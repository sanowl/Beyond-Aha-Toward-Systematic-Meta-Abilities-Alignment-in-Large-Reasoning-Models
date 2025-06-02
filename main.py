import torch
from torch.nn import CrossEntropyLoss
import math
from collections import OrderedDict
from typing import Dict, Any, Type, List, Tuple, Optional
import os
import re
import argparse
import json
import subprocess
import yaml

from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig,
    TrainingArguments,
    GenerationConfig
)
from datasets import Dataset, load_dataset
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from peft import LoraConfig, PeftModel
import secrets

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = "pipeline_checkpoints_advanced"
MERGED_MODEL_OUTPUT_DIR_BASE = "merged_models_output"
FULL_MODEL_EXPORT_DIR_BASE = "exported_full_models"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(MERGED_MODEL_OUTPUT_DIR_BASE, exist_ok=True)
os.makedirs(FULL_MODEL_EXPORT_DIR_BASE, exist_ok=True)

class DeductionTaskGenerator:
    VARS = "pqrstuvwxyz"
    def __init__(self, num_vars: int = 3, num_clauses: int = 4): 
        self.num_vars = num_vars
        self.num_clauses = num_clauses
        self.vars = self.VARS[:num_vars]

    def _random_literal(self) -> str:
        v = secrets.choice(self.vars)
        return v if secrets.SystemRandom().random() < 0.5 else f"¬{v}" 

    def _random_clause(self) -> str:
        k = secrets.SystemRandom().randint(2, max(2, self.num_vars -1)) 
        lits = " ∨ ".join(sorted(list(set(self._random_literal() for _ in range(k))))) 
        return f"({lits})"

    def sample(self) -> Dict[str, str]: 
        clauses_conj = " ∧ ".join(self._random_clause() for _ in range(self.num_clauses))
        prompt_text = (
            "<task_type>[Deduction]</task_type>\n"
            "<problem>Given the Boolean formula: " + clauses_conj + "\n"
            "Provide a satisfying truth assignment (e.g., p=true, q=false, ...) or state UNSAT.</problem>\n"
            "<思考>" 
        )
        gt_answer = "UNSAT" 
        return {"query": prompt_text, "ground_truth_answer": gt_answer, "task_type": "Deduction"}

class InductionTaskGenerator:
    def sample(self) -> Dict[str, str]:
        start = secrets.SystemRandom().randint(1, 10)
        diff = secrets.SystemRandom().randint(1, 5) * secrets.choice([-1, 1])
        length = 5
        seq = [start + i * diff for i in range(length)]
        mask_idx = secrets.SystemRandom().randint(0, length - 1)
        gt_answer = str(seq[mask_idx])
        seq_str_parts = ["__" if i == mask_idx else str(val) for i, val in enumerate(seq)]
        prompt_text = (
            "<task_type>[Induction]</task_type>\n"
            "<problem>Identify the pattern and fill in the missing value marked by __ in the sequence: "
            f"{', '.join(seq_str_parts)}</problem>\n"
            "<思考>"
        )
        return {"query": prompt_text, "ground_truth_answer": gt_answer, "task_type": "Induction"}

class AbductionTaskGenerator:
    def sample(self) -> Dict[str, str]:
        target = secrets.SystemRandom().randint(5, 20)
        a = secrets.SystemRandom().randint(1, target - 1)
        b = target - a 
        while b <= 0 or b == a : 
            a = secrets.SystemRandom().randint(1, target - 1)
            b = target - a
            if target <=2: a,b = 1,1; break
        prompt_text = (
            "<task_type>[Abduction]</task_type>\n"
            f"<problem>Given the observation that two hidden positive integers sum to {target}, "
            "provide one plausible pair of such integers (e.g., X + Y = target).</problem>\n"
            "<思考>"
        )
        gt_answer = f"{min(a,b)} + {max(a,b)}" 
        return {"query": prompt_text, "ground_truth_answer": gt_answer, "task_type": "Abduction"}

ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
THINK_RE = re.compile(r"<思考>([\s\S]*?)</思考>", re.IGNORECASE | re.DOTALL) 

def format_reward_tensor(decoded_texts: List[str]) -> torch.Tensor:
    rewards = []
    for text in decoded_texts:
        has_think = bool(THINK_RE.search(text))
        has_answer = bool(ANSWER_RE.search(text))
        rewards.append(1.0 if has_think and has_answer else -1.0)
    return torch.tensor(rewards, dtype=torch.float32).to(DEVICE)

def answer_reward_tensor(decoded_texts: List[str], ground_truth_answers: List[str], task_types: List[str]) -> torch.Tensor:
    rewards = []
    for text, gt_answer, task_type in zip(decoded_texts, ground_truth_answers, task_types):
        match = ANSWER_RE.search(text)
        current_reward = -2.0
        if match:
            extracted_answer = match.group(1).strip()
            is_correct = False
            if task_type == "Deduction": is_correct = (extracted_answer.upper() == gt_answer.upper())
            elif task_type == "Induction": is_correct = (extracted_answer == gt_answer)
            elif task_type == "Abduction":
                try:
                    parts_gt = sorted([int(p.strip()) for p in gt_answer.split('+')])
                    parts_ex = sorted([int(p.strip()) for p in extracted_answer.split('+')])
                    is_correct = (parts_gt == parts_ex) and (sum(parts_ex) == sum(parts_gt))
                except: is_correct = (extracted_answer == gt_answer)
            if is_correct: current_reward = 2.0
        rewards.append(current_reward)
    return torch.tensor(rewards, dtype=torch.float32).to(DEVICE)

def calculate_total_reward_tensor(decoded_texts: List[str], ground_truth_answers: List[str], task_types: List[str]) -> torch.Tensor:
    r_format = format_reward_tensor(decoded_texts)
    r_answer = answer_reward_tensor(decoded_texts, ground_truth_answers, task_types)
    return r_format + r_answer

class MetaAbilityPPOTrainer:
    def __init__(self,
                 model_name_or_path: str,
                 task_name: str,
                 task_generator: Any,
                 ppo_config_args: Dict[str, Any],
                 num_samples_total: int = 128, 
                 output_dir_suffix: str = "_specialist_ppo",
                 use_lora: bool = True):

        self.task_name = task_name
        self.task_generator = task_generator
        self.num_samples_total = num_samples_total
        self.output_dir = os.path.join(CHECKPOINT_DIR, f"{self.task_name.lower()}{output_dir_suffix}")
        self.base_model_name_or_path = model_name_or_path
        self.use_lora = use_lora
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 
        )

        lora_config = None
        if self.use_lora:
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            print("Tokenizer pad_token set to eos_token.")

        model = AutoModelForCausalLMWithValueHead.from_pretrained(
            model_name_or_path,
            quantization_config=quantization_config,
            device_map={"": DEVICE}, 
            peft_config=lora_config if self.use_lora else None,
            torch_dtype=torch.bfloat16 
        )
        
        ppo_config = PPOConfig(**ppo_config_args)

        self.ppo_trainer = PPOTrainer(
            config=ppo_config,
            model=model,
            ref_model=None, 
            tokenizer=self.tokenizer,
            dataset=None 
        )
        
        self.generation_kwargs = {
            "min_length": -1, 
            "top_k": 0.0,
            "top_p": 1.0,
            "do_sample": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "max_new_tokens": 50 
        }
        print(f"PPO Trainer initialized for {self.task_name} on device {DEVICE}")


    def train(self):
        print(f"\n--- Stage A: PPO Training {self.task_name} specialist ---")
        
        num_batches = self.num_samples_total // self.ppo_trainer.config.batch_size
        if num_batches == 0:
            print(f"Warning: num_samples_total ({self.num_samples_total}) is less than batch_size ({self.ppo_trainer.config.batch_size}). Training for 1 batch.")
            num_batches = 1

        for batch_idx in range(num_batches):
            print(f"  PPO Batch {batch_idx + 1}/{num_batches} for {self.task_name}")
            
            batch_data = [self.task_generator.sample() for _ in range(self.ppo_trainer.config.batch_size)]
            
            query_texts = [sample["query"] for sample in batch_data]
            ground_truth_answers = [sample["ground_truth_answer"] for sample in batch_data]
            task_types = [sample["task_type"] for sample in batch_data]

            query_tensors = [self.tokenizer.encode(q, return_tensors="pt").to(DEVICE).squeeze(0) for q in query_texts]
            
            response_tensors = self.ppo_trainer.generate(query_tensors, **self.generation_kwargs)
            
            response_texts = []
            for i in range(len(response_tensors)):
                query_len = query_tensors[i].shape[0]
                actual_response_tensor = response_tensors[i].squeeze()
                if actual_response_tensor.shape[0] > query_len:
                     response_texts.append(self.tokenizer.decode(actual_response_tensor[query_len:], skip_special_tokens=True))
                else: 
                    response_texts.append("")

            rewards = calculate_total_reward_tensor(response_texts, ground_truth_answers, task_types)
            
            stats = self.ppo_trainer.step(query_tensors, [r.squeeze() for r in response_tensors], rewards.tolist()) 
            self.ppo_trainer.log_stats(stats, {"query": query_texts, "response": response_texts}, rewards)
            print(f"    Batch {batch_idx+1} PPO step completed. Mean reward: {torch.mean(rewards).item():.2f}")

        print(f"--- {self.task_name} specialist PPO training finished ---")
        self.ppo_trainer.save_model(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir) 
        print(f"  {self.task_name} specialist adapters/model saved to {self.output_dir}")
        
        if self.use_lora:
            full_model_export_path = os.path.join(FULL_MODEL_EXPORT_DIR_BASE, f"{self.task_name.lower()}_specialist_full")
            print(f"  Exporting LoRA specialist {self.task_name} to full model at {full_model_export_path}...")
            export_lora_to_full_model(self.base_model_name_or_path, self.output_dir, full_model_export_path)
            return full_model_export_path 
        else:
            return self.output_dir # Path to the full fine-tuned model

def export_lora_to_full_model(base_model_name_or_path: str, lora_adapter_path: str, full_model_output_path: str):
    print(f"  Loading base model '{base_model_name_or_path}' for LoRA merge...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name_or_path,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        device_map="auto" 
    )
    tokenizer = AutoTokenizer.from_pretrained(lora_adapter_path) # Adapters dir should have tokenizer

    print(f"  Loading PeftModel with adapters from '{lora_adapter_path}'...")
    peft_model = PeftModel.from_pretrained(base_model, lora_adapter_path)
    
    print("  Merging LoRA adapters into the base model...")
    merged_model = peft_model.merge_and_unload()
    
    print(f"  Saving full merged model to '{full_model_output_path}'...")
    os.makedirs(full_model_output_path, exist_ok=True)
    merged_model.save_pretrained(full_model_output_path)
    tokenizer.save_pretrained(full_model_output_path)
    print(f"  Full model exported successfully to {full_model_output_path}.")


def merge_models_with_mergekit(
    specialist_full_model_paths: Dict[str, str], 
    lambdas: Dict[str, float],
    base_model_for_mergekit_config: str, 
    output_model_name: str = "merged_lrm_mergekit"
) -> Optional[str]:
    print("\n--- Stage B: Starting Parameter-Space Merging with mergekit ---")
    
    mergekit_config = {
        "base_model": base_model_for_mergekit_config, 
        "models": [], # Changed from "slices" to "models" for newer mergekit syntax with "parameters"
        "merge_method": "linear", 
        "dtype": "bfloat16" 
    }

    for specialist_key, model_path in specialist_full_model_paths.items():
        if specialist_key in lambdas and os.path.isdir(model_path): # Ensure path is valid
             mergekit_config["models"].append({ # Changed from "slices"
                 "model": os.path.abspath(model_path), 
                 "parameters": {"weight": lambdas[specialist_key]} # Parameters for the model
             })
        else:
            print(f"  Warning: No lambda weight found for specialist '{specialist_key}' or path invalid '{model_path}'. Skipping.")

    if not mergekit_config["models"]:
        print("  Error: No specialist models to merge. Aborting mergekit.")
        return None

    merged_output_path = os.path.join(MERGED_MODEL_OUTPUT_DIR_BASE, output_model_name)
    # mergekit creates the directory, ensure parent exists
    os.makedirs(os.path.dirname(merged_output_path), exist_ok=True)


    config_filename = f"mergekit_config_{output_model_name}.yml"
    with open(config_filename, 'w') as f:
        yaml.dump(mergekit_config, f, sort_keys=False)
    
    print(f"  Generated mergekit config: {config_filename}")
    print(f"  Mergekit config content:\n{yaml.dump(mergekit_config)}")

    try:
        command = [
            "mergekit-yaml", 
            config_filename, 
            merged_output_path,
            "--cuda", 
            "--allow-crimes", 
        ]
        print(f"  Executing mergekit command: {' '.join(command)}")
        
        process = subprocess.run(command, capture_output=True, text=True, check=True)
        print("  mergekit stdout:")
        print(process.stdout)
        if process.stderr:
            print("  mergekit stderr:")
            print(process.stderr)
        
        print(f"--- Mergekit operation completed. Merged model should be in {merged_output_path} ---")
        os.remove(config_filename) 
        return merged_output_path
    except subprocess.CalledProcessError as e:
        print(f"  Error during mergekit execution: {e}")
        print("  mergekit stdout (error):")
        print(e.stdout)
        print("  mergekit stderr (error):")
        print(e.stderr)
        return None
    except FileNotFoundError:
        print("  Error: mergekit-yaml command not found. Is mergekit installed and in PATH?")
        return None

class DomainPPOTrainer(MetaAbilityPPOTrainer): 
    def __init__(self,
                 merged_model_path: str, 
                 domain_dataset_name_or_path: str, 
                 ppo_config_args: Dict[str, Any],
                 num_samples_total: int = 256,
                 output_dir_suffix: str = "_domain_rl_ppo",
                 use_lora: bool = True): 

        self.domain_dataset_name = domain_dataset_name_or_path
        base_merged_name = os.path.basename(merged_model_path.strip(os.sep)) 
        self.output_dir = os.path.join(CHECKPOINT_DIR, f"{base_merged_name}{output_dir_suffix}")
        self.base_model_name_or_path = merged_model_path # For LoRA export if needed
        self.use_lora = use_lora

        quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
        lora_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM") if self.use_lora else None
        
        self.tokenizer = AutoTokenizer.from_pretrained(merged_model_path) 
        if self.tokenizer.pad_token is None: self.tokenizer.pad_token = self.tokenizer.eos_token

        model = AutoModelForCausalLMWithValueHead.from_pretrained(
            merged_model_path, 
            quantization_config=quantization_config,
            device_map={"": DEVICE},
            peft_config=lora_config,
            torch_dtype=torch.bfloat16
        )
        
        ppo_config = PPOConfig(**ppo_config_args)
        self.ppo_trainer = PPOTrainer(ppo_config, model, ref_model=None, tokenizer=self.tokenizer)
        
        self.generation_kwargs = {"min_length": -1, "top_k": 0.0, "top_p": 1.0, "do_sample": True,
                                  "pad_token_id": self.tokenizer.pad_token_id, "eos_token_id": self.tokenizer.eos_token_id,
                                  "max_new_tokens": 60} 
        
        self.num_samples_total = num_samples_total
        print(f"Domain PPO Trainer initialized for dataset {self.domain_dataset_name} on device {DEVICE}")

    def load_and_prepare_dataset(self):
        print(f"  Loading placeholder domain dataset: {self.domain_dataset_name}")
        if self.domain_dataset_name == "dummy_math":
            data = []
            for _ in range(self.num_samples_total):
                n1, n2 = secrets.SystemRandom().randint(0,100), secrets.SystemRandom().randint(0,100)
                op_choice = secrets.choice(['+', '-', '*'])
                if op_choice == '+': query, answer = f"What is {n1} + {n2}?", str(n1+n2)
                elif op_choice == '-': query, answer = f"What is {n1} - {n2}?", str(n1-n2)
                else: query, answer = f"What is {n1} * {n2}?", str(n1*n2)
                data.append({"query": f"<task_type>[Math]</task_type>\n<problem>{query}</problem>\n<思考>", 
                             "ground_truth_answer": answer})
            self.dataset = Dataset.from_list(data)
        else:
            raise ValueError(f"Unknown or unimplemented domain dataset: {self.domain_dataset_name}")
        print(f"  Loaded {len(self.dataset)} samples for domain {self.domain_dataset_name}")


    def train_domain(self): 
        print(f"\n--- Stage C: Domain-Specific PPO RL Training ({self.domain_dataset_name}) ---")
        self.load_and_prepare_dataset()

        num_batches = len(self.dataset) // self.ppo_trainer.config.batch_size
        if num_batches == 0: num_batches = 1
        
        current_sample_idx = 0
        for batch_idx in range(num_batches):
            print(f"  PPO Batch {batch_idx + 1}/{num_batches} for domain {self.domain_dataset_name}")
            
            batch_end_idx = min(current_sample_idx + self.ppo_trainer.config.batch_size, len(self.dataset))
            if current_sample_idx >= batch_end_idx: break 

            batch_data = self.dataset[current_sample_idx:batch_end_idx]
            current_sample_idx = batch_end_idx
            
            query_texts = batch_data["query"]
            ground_truth_answers = batch_data["ground_truth_answer"]
            task_types = ["DomainMath"] * len(query_texts) 

            query_tensors = [self.tokenizer.encode(q, return_tensors="pt").to(DEVICE).squeeze(0) for q in query_texts]
            response_tensors = self.ppo_trainer.generate(query_tensors, **self.generation_kwargs)
            
            response_texts = []
            for i in range(len(response_tensors)):
                query_len = query_tensors[i].shape[0]
                actual_response_tensor = response_tensors[i].squeeze()
                if actual_response_tensor.shape[0] > query_len:
                     response_texts.append(self.tokenizer.decode(actual_response_tensor[query_len:], skip_special_tokens=True))
                else: response_texts.append("")

            rewards_list = []
            for resp_text, gt_ans in zip(response_texts, ground_truth_answers):
                match = ANSWER_RE.search(resp_text)
                reward = 0.0
                if match:
                    extracted_answer = match.group(1).strip()
                    if extracted_answer == gt_ans: reward = 1.0
                rewards_list.append(reward)
            rewards = torch.tensor(rewards_list, dtype=torch.float32).to(DEVICE)
            
            stats = self.ppo_trainer.step(query_tensors, [r.squeeze() for r in response_tensors], rewards.tolist())
            self.ppo_trainer.log_stats(stats, {"query": query_texts, "response": response_texts}, rewards)
            print(f"    Batch {batch_idx+1} Domain PPO step completed. Mean reward: {torch.mean(rewards).item():.2f}")

        print(f"--- Domain-Specific PPO RL training finished ---")
        self.ppo_trainer.save_model(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        print(f"  Domain RL adapters/model saved to {self.output_dir}")

        if self.use_lora:
            full_model_export_path = os.path.join(FULL_MODEL_EXPORT_DIR_BASE, f"{os.path.basename(self.output_dir)}_full")
            print(f"  Exporting LoRA domain model to full model at {full_model_export_path}...")
            export_lora_to_full_model(self.base_model_name_or_path, self.output_dir, full_model_export_path)
            return full_model_export_path
        else:
            return self.output_dir


def evaluate_pass_at_1(
    model_path_or_name: str, 
    eval_dataset_name: str, 
    num_eval_samples: int = 50, 
    is_lora_output: bool = False, # If the model_path_or_name is a LoRA adapter dir
    base_model_for_lora_eval: Optional[str] = None # Needed if is_lora_output is True
) -> float:
    print(f"\n--- Evaluating pass@1 for model: {model_path_or_name} on {eval_dataset_name} ---")
    
    model_to_eval = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path_or_name if not is_lora_output else base_model_for_lora_eval) # Load tokenizer from base if evaluating LoRA directly
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
        
        quant_config_eval = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)

        if is_lora_output:
            if not base_model_for_lora_eval:
                raise ValueError("base_model_for_lora_eval must be provided if is_lora_output is True.")
            print(f"  Loading LoRA model for evaluation. Base: {base_model_for_lora_eval}, Adapters: {model_path_or_name}")
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_for_lora_eval,
                quantization_config=quant_config_eval,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            model_to_eval = PeftModel.from_pretrained(base_model, model_path_or_name).eval()
        else:
            print(f"  Loading full model for evaluation: {model_path_or_name}")
            model_to_eval = AutoModelForCausalLM.from_pretrained(
                model_path_or_name,
                quantization_config=quant_config_eval,
                device_map="auto", 
                torch_dtype=torch.bfloat16
            ).eval()

    except Exception as e:
        print(f"  Error loading model for evaluation: {e}. Skipping evaluation.")
        return 0.0

    eval_data = []
    if eval_dataset_name == "dummy_eval_math":
        for i in range(num_eval_samples):
            n1, n2 = secrets.SystemRandom().randint(0, 20), secrets.SystemRandom().randint(0, 20)
            query = f"<task_type>[Math]</task_type>\n<problem>Calculate {n1} * {n2}.</problem>\n<思考>"
            gt = str(n1 * n2)
            eval_data.append({"query": query, "ground_truth_answer": gt})
    else:
        print(f"  Warning: Unknown evaluation dataset '{eval_dataset_name}'. Using dummy math.")
        for i in range(num_eval_samples):
            n1, n2 = secrets.SystemRandom().randint(0, 20), secrets.SystemRandom().randint(0, 20)
            query = f"<task_type>[Math]</task_type>\n<problem>Calculate {n1} * {n2}.</problem>\n<思考>"
            gt = str(n1 * n2)
            eval_data.append({"query": query, "ground_truth_answer": gt})


    correct_count = 0
    generation_config_eval = GenerationConfig(
        max_new_tokens=50,
        do_sample=False, 
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        num_beams=1 
    )

    for i, item in enumerate(eval_data):
        print(f"  Evaluating sample {i+1}/{num_eval_samples}...", end='\r')
        inputs = tokenizer(item["query"], return_tensors="pt").to(model_to_eval.device) # Ensure inputs are on model's device
        
        with torch.no_grad():
            outputs = model_to_eval.generate(**inputs, generation_config=generation_config_eval)
        
        query_len = inputs.input_ids.shape[1]
        decoded_output = tokenizer.decode(outputs[0, query_len:], skip_special_tokens=True)
        
        match = ANSWER_RE.search(decoded_output)
        if match:
            extracted_answer = match.group(1).strip()
            if extracted_answer == item["ground_truth_answer"]:
                correct_count += 1
    
    pass_at_1 = (correct_count / num_eval_samples) * 100 if num_eval_samples > 0 else 0.0
    print(f"\n  Evaluation completed. Correct: {correct_count}/{num_eval_samples}. Pass@1: {pass_at_1:.2f}% for {model_path_or_name}")
    return pass_at_1


def main():
    parser = argparse.ArgumentParser(description="Advanced Meta-Ability Alignment Pipeline (Hu et al., 2025)")
    parser.add_argument("stage", choices=["stage_a", "stage_b", "stage_c", "full_pipeline", "evaluate"], 
                        help="Which stage of the pipeline to run or 'evaluate' a model.")
    
    parser.add_argument("--base_model_name", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0", help="Base Hugging Face model name for Stage A and for mergekit's base.")
    parser.add_argument("--eval_dataset", type=str, default="dummy_eval_math", help="Evaluation dataset name.")
    parser.add_argument("--num_eval_samples", type=int, default=20, help="Number of samples for pass@1 evaluation.")

    parser.add_argument("--task_type_a", choices=["deduction", "induction", "abduction"], help="Task type for Stage A.")
    parser.add_argument("--stage_a_samples", type=int, default=64, help="Total samples for Stage A PPO training per specialist.")
    parser.add_argument("--stage_a_batch_size", type=int, default=4, help="Batch size for Stage A PPO.")
    parser.add_argument("--stage_a_lr", type=float, default=1.41e-5, help="Learning rate for Stage A PPO.")
    parser.add_argument("--no_lora_stage_a", action="store_true", help="Disable LoRA for Stage A training.")

    parser.add_argument("--deduction_model_path", type=str, default=os.path.join(FULL_MODEL_EXPORT_DIR_BASE, "deduction_specialist_full"), help="Path to FULL deduction specialist model directory for mergekit.")
    parser.add_argument("--induction_model_path", type=str, default=os.path.join(FULL_MODEL_EXPORT_DIR_BASE, "induction_specialist_full"), help="Path to FULL induction specialist model directory for mergekit.")
    parser.add_argument("--abduction_model_path", type=str, default=os.path.join(FULL_MODEL_EXPORT_DIR_BASE, "abduction_specialist_full"), help="Path to FULL abduction specialist model directory for mergekit.")
    parser.add_argument("--lambda_d", type=float, default=1.0)
    parser.add_argument("--lambda_i", type=float, default=0.2)
    parser.add_argument("--lambda_a", type=float, default=0.1)
    parser.add_argument("--merged_model_name_b", type=str, default="merged_lrm_mk", help="Output name for mergekit model.")

    parser.add_argument("--merged_model_path_for_c", type=str, default=os.path.join(MERGED_MODEL_OUTPUT_DIR_BASE, "merged_lrm_mk"), help="Path to merged model directory for Stage C.")
    parser.add_argument("--domain_dataset_c", type=str, default="dummy_math", help="Domain dataset for Stage C.")
    parser.add_argument("--stage_c_samples", type=int, default=128, help="Total samples for Stage C PPO training.")
    parser.add_argument("--stage_c_batch_size", type=int, default=4, help="Batch size for Stage C PPO.")
    parser.add_argument("--stage_c_lr", type=float, default=5e-6, help="Learning rate for Stage C PPO.")
    parser.add_argument("--no_lora_stage_c", action="store_true", help="Disable LoRA for Stage C training.")

    parser.add_argument("--eval_model_path", type=str, help="Path to model directory to evaluate.")
    parser.add_argument("--eval_is_lora_output", action="store_true", help="Is the model to evaluate a LoRA adapter directory (requires base_model_name for eval)?")


    args = parser.parse_args()
    print(f"Running on device: {DEVICE}")

    ppo_config_args_stage_a = {
        "batch_size": args.stage_a_batch_size,
        "learning_rate": args.stage_a_lr,
        "log_with": None, 
        "mini_batch_size": args.stage_a_batch_size // 2 if args.stage_a_batch_size > 1 else 1,
        "gradient_accumulation_steps": 1,
        "ppo_epochs": 4, 
    }
    ppo_config_args_stage_c = {
        "batch_size": args.stage_c_batch_size,
        "learning_rate": args.stage_c_lr,
        "log_with": None,
        "mini_batch_size": args.stage_c_batch_size // 2 if args.stage_c_batch_size > 1 else 1,
        "gradient_accumulation_steps": 1,
        "ppo_epochs": 4,
    }

    ded_model_full_path_for_b = args.deduction_model_path
    ind_model_full_path_for_b = args.induction_model_path
    abd_model_full_path_for_b = args.abduction_model_path
    
    merged_model_path_for_stage_c = args.merged_model_path_for_c


    if args.stage == "stage_a" or args.stage == "full_pipeline":
        task_types_to_run_a = [args.task_type_a] if args.stage == "stage_a" and args.task_type_a else ["deduction", "induction", "abduction"]
        
        for task_name in task_types_to_run_a:
            generator_class = None
            if task_name == "deduction": generator_class = DeductionTaskGenerator()
            elif task_name == "induction": generator_class = InductionTaskGenerator()
            elif task_name == "abduction": generator_class = AbductionTaskGenerator()
            
            if generator_class:
                use_lora_a = not args.no_lora_stage_a
                trainer_a = MetaAbilityPPOTrainer(
                    args.base_model_name, task_name, generator_class, 
                    ppo_config_args_stage_a, num_samples_total=args.stage_a_samples,
                    use_lora=use_lora_a
                )
                output_model_path_a = trainer_a.train() # This is LoRA adapter path or full model path
                evaluate_pass_at_1(
                    output_model_path_a, 
                    args.eval_dataset, 
                    args.num_eval_samples,
                    is_lora_output=use_lora_a,
                    base_model_for_lora_eval=args.base_model_name if use_lora_a else None
                )
                
                if task_name == "deduction": ded_model_full_path_for_b = output_model_path_a if not use_lora_a else os.path.join(FULL_MODEL_EXPORT_DIR_BASE, f"{task_name.lower()}_specialist_full")
                elif task_name == "induction": ind_model_full_path_for_b = output_model_path_a if not use_lora_a else os.path.join(FULL_MODEL_EXPORT_DIR_BASE, f"{task_name.lower()}_specialist_full")
                elif task_name == "abduction": abd_model_full_path_for_b = output_model_path_a if not use_lora_a else os.path.join(FULL_MODEL_EXPORT_DIR_BASE, f"{task_name.lower()}_specialist_full")

            else:
                print(f"Unknown task type for Stage A: {task_name}")


    if args.stage == "stage_b" or args.stage == "full_pipeline":
        specialist_paths_b = {
            "deduction": ded_model_full_path_for_b, 
            "induction": ind_model_full_path_for_b, 
            "abduction": abd_model_full_path_for_b
        }
        lambdas_b = {
            "deduction": args.lambda_d, "induction": args.lambda_i, "abduction": args.lambda_a
        }
        
        all_specialists_exist_b = True
        for name, path in specialist_paths_b.items():
            if not os.path.isdir(path): 
                print(f"Error: Specialist model directory for {name} ('{path}') not found. Run Stage A (and export if LoRA) or provide correct paths.")
                all_specialists_exist_b = False
        
        if all_specialists_exist_b:
            merged_output_dir_b = merge_models_with_mergekit(
                specialist_paths_b, lambdas_b, 
                base_model_for_mergekit_config=args.base_model_name, 
                output_model_name=args.merged_model_name_b
            )
            if merged_output_dir_b:
                evaluate_pass_at_1(merged_output_dir_b, args.eval_dataset, args.num_eval_samples)
                merged_model_path_for_stage_c = merged_output_dir_b 
            else:
                print("Skipping further steps as mergekit failed.")
                if args.stage == "full_pipeline": return
        else:
            print("Skipping Stage B due to missing specialist model directories.")
            if args.stage == "full_pipeline": return


    if args.stage == "stage_c" or args.stage == "full_pipeline":
        if not os.path.isdir(merged_model_path_for_stage_c):
            print(f"Error: Merged model directory '{merged_model_path_for_stage_c}' not found. Run Stage B or provide correct path.")
            if args.stage == "full_pipeline": return
        else:
            use_lora_c = not args.no_lora_stage_c
            trainer_c = DomainPPOTrainer(
                merged_model_path_for_stage_c, args.domain_dataset_c, 
                ppo_config_args_stage_c, num_samples_total=args.stage_c_samples,
                use_lora=use_lora_c
            )
            output_model_path_c = trainer_c.train_domain() # LoRA adapter path or full model path
            
            evaluate_pass_at_1(
                output_model_path_c, 
                args.eval_dataset, 
                args.num_eval_samples,
                is_lora_output=use_lora_c,
                # If LoRA was used in Stage C, the base for these adapters is the merged model from Stage B
                base_model_for_lora_eval=merged_model_path_for_stage_c if use_lora_c else None
            )
            
    if args.stage == "evaluate":
        if not args.eval_model_path:
            parser.error("--eval_model_path is required for 'evaluate' stage.")
        if not os.path.isdir(args.eval_model_path):
             print(f"Error: Model path for evaluation '{args.eval_model_path}' not found or not a directory.")
        else:
            evaluate_pass_at_1(
                args.eval_model_path, 
                args.eval_dataset, 
                args.num_eval_samples,
                is_lora_output=args.eval_is_lora_output,
                base_model_for_lora_eval=args.base_model_name if args.eval_is_lora_output else None # Assuming base_model_name is the original base if eval_is_lora_output
            )

    print("\nPipeline execution finished.")

if __name__ == "__main__":
    main()
