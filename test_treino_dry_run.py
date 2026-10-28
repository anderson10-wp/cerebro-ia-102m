#!/usr/bin/env python3
"""
Teste de dry-run do pipeline de treinamento massivo R1
"""
import unittest
import sys
import torch
import torch.nn.functional as F
from pathlib import Path

WORKSPACE = Path(r"C:\Users\Fernando\.gemini\antigravity\scratch\cerebro-ia-102m")
sys.path.insert(0, str(WORKSPACE))

from tokenizers import Tokenizer
from modelo_synapse_1b import BrainMoE1B, Config1B

class TestTreinoDryRun(unittest.TestCase):
    def setUp(self):
        self.tok = Tokenizer.from_file(str(WORKSPACE / "tokenizer.json"))
        # Configuração mini para dry-run ultrarrápido
        self.mini_cfg = Config1B(
            vocab_size=self.tok.get_vocab_size(),
            dim=64,
            layers=2,
            n_heads=2,
            head_dim=32,
            hidden_dim=128,
            num_experts=4,
            top_k=2,
            context=256
        )
        self.model = BrainMoE1B(self.mini_cfg)

    def test_masked_dataset_and_loss(self):
        from alinhar_synapse_1b_r1_massivo import MaskedReasoningDataset
        
        # Cria arquivo temporário JSONL de teste
        import tempfile, json
        with tempfile.NamedTemporaryFile("w+", suffix=".jsonl", delete=False, encoding="utf-8") as f:
            f.write(json.dumps({
                "pergunta": "Quanto é 2 + 2?",
                "pensamento": "[Entendimento]: Operação básica.\n[Auto-Verificação]: Prova real.\n[Síntese]: 4.",
                "resposta": "4"
            }) + "\n")
            f.write(json.dumps({
                "pergunta": "Olá!",
                "pensamento": "",
                "resposta": "Olá! Tudo bem?"
            }) + "\n")
            tmp_path = f.name
            
        ds = MaskedReasoningDataset(tmp_path, self.tok, max_len=256)
        self.assertEqual(len(ds), 2)
        
        inp_ids, labels = ds[0]
        self.assertEqual(inp_ids.shape[0], 256)
        self.assertEqual(labels.shape[0], 256)
        
        # Verifica se os primeiros tokens (prompt) receberam -100
        self.assertEqual(labels[0].item(), -100)
        
        # Teste de 1 passo de forward e backward
        batch_x = inp_ids.unsqueeze(0)
        batch_y = labels.unsqueeze(0)
        
        logits, aux_loss = self.model(batch_x)
        self.assertEqual(logits.shape, (1, 256, self.mini_cfg.vocab_size))
        
        # Shift logits e labels para autoregressão
        shift_logits = logits[..., :-1, :].contiguous().view(-1, self.mini_cfg.vocab_size)
        shift_labels = batch_y[..., 1:].contiguous().view(-1)
        
        loss_ce = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)
        total_loss = loss_ce + 0.01 * aux_loss
        
        self.assertTrue(torch.isfinite(total_loss))
        total_loss.backward()
        
        # Limpa arquivo temporário
        Path(tmp_path).unlink(missing_ok=True)

if __name__ == "__main__":
    unittest.main()
