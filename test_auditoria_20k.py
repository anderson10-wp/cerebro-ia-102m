#!/usr/bin/env python3
"""
Testes unitários para o script de auditoria estatística do dataset 20k
"""
import unittest
import sys
from pathlib import Path

WORKSPACE = Path(r"C:\Users\Fernando\.gemini\antigravity\scratch\cerebro-ia-102m")
sys.path.insert(0, str(WORKSPACE))

class TestAuditoriaDataset20k(unittest.TestCase):
    def test_classificacao_dominio(self):
        from auditar_dataset_20k import classificar_dominio
        
        # Teste Matemática
        dom_math = classificar_dominio("Resolva a equação 2x + 4 = 10", "Cálculo da álgebra")
        self.assertEqual(dom_math, "Matemática & Cálculo")
        
        # Teste Código
        dom_code = classificar_dominio("Como criar uma função def soma(a, b) em Python?", "Código Python")
        self.assertEqual(dom_code, "Algoritmos & Código")
        
        # Teste Direito
        dom_dir = classificar_dominio("Qual a diferença entre dolo eventual e culpa consciente no direito penal?", "Artigo 18 do Código Penal")
        self.assertEqual(dom_dir, "Direito & Brasil")
        
        # Teste Identidade
        dom_id = classificar_dominio("Quem é você e quem é o Nandinho?", "Synapse criada pelo Nandinho")
        self.assertEqual(dom_id, "Identidade Synapse")
        
        # Teste Direta
        dom_dir = classificar_dominio("Bom dia!", "")
        self.assertEqual(dom_dir, "Conversa Direta")

if __name__ == "__main__":
    unittest.main()
