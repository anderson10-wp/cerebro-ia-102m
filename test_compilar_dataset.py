#!/usr/bin/env python3
"""
Testes de integridade do pipeline de compilação do dataset R1 20k
"""
import os
import sys
import json
import unittest
from pathlib import Path

WORKSPACE = Path(r"C:\Users\Fernando\.gemini\antigravity\scratch\cerebro-ia-102m")
sys.path.insert(0, str(WORKSPACE))

try:
    from tokenizers import Tokenizer
except ImportError:
    Tokenizer = None

class TestCompilarDatasetR1(unittest.TestCase):
    def setUp(self):
        self.tok_path = WORKSPACE / "tokenizer.json"
        self.assertTrue(self.tok_path.exists(), "tokenizer.json deve existir no workspace")
        if Tokenizer:
            self.tok = Tokenizer.from_file(str(self.tok_path))
        else:
            self.tok = None

    def test_sample_validation_logic(self):
        """Verifica se a lógica de validação aceita amostras válidas e rejeita inválidas."""
        from compilar_dataset_r1_20k import validar_e_formatar_amostra
        
        # Amostra válida com raciocínio
        valida = {
            "pergunta": "Quanto é 12 + 15?",
            "pensamento": "[Entendimento]: Soma de inteiros.\n[Desenvolvimento / Cálculo]: 12 + 15 = 27.\n[Auto-Verificação]: 27 - 12 = 15.\n[Síntese]: Resultado 27.",
            "resposta": "12 + 15 = 27."
        }
        res = validar_e_formatar_amostra(valida, self.tok, max_len=1024)
        self.assertIsNotNone(res)
        self.assertIn("pergunta", res)
        self.assertIn("pensamento", res)
        self.assertIn("resposta", res)
        
        # Amostra inválida (vazia)
        invalida = {"pergunta": "", "pensamento": "...", "resposta": "..."}
        self.assertIsNone(validar_e_formatar_amostra(invalida, self.tok))
        
        # Amostra longa que ultrapassa max_len deve ser descartada (retornar None)
        longa = {
            "pergunta": "X " * 600,
            "pensamento": "Y " * 600,
            "resposta": "Z " * 600
        }
        self.assertIsNone(validar_e_formatar_amostra(longa, self.tok, max_len=1024))

    def test_direct_response_support(self):
        """Verifica se saudações diretas (anti-catastrophic forgetting) são suportadas."""
        from compilar_dataset_r1_20k import validar_e_formatar_amostra
        
        direta = {
            "pergunta": "Olá, tudo bem?",
            "pensamento": "",
            "resposta": "Olá! Tudo ótimo, como posso ajudar você hoje?"
        }
        res = validar_e_formatar_amostra(direta, self.tok, max_len=1024, permitir_sem_think=True)
        self.assertIsNotNone(res)
        self.assertEqual(res["pensamento"], "")

if __name__ == "__main__":
    unittest.main()
