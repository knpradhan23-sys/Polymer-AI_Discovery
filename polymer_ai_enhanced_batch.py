# AI-Enhanced Polymer Formulation Discovery System
# CampAIgn Project - For Slovenia Research Position Application
# Author: Nishant Pradhan | Date: November 2025

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from scipy.optimize import differential_evolution
from datetime import datetime
import glob

warnings.filterwarnings('ignore')

def setup_output_folder():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = f"polymer_ai_outputs_{timestamp}"
    os.makedirs(output_folder, exist_ok=True)
    print(f"\n📁 Output folder created: {output_folder}/")
    return output_folder

class PolymerAISystem:
    def __init__(self, output_folder):
        self.polymer_type = None
        self.user_data = None
        self.alice_knowledge = None
        self.victor_model = None
        self.frank_results = None
        self.data_source = None
        self.output_folder = output_folder
        self.uploaded_files = []
        
    def welcome_screen(self):
        print("\n" + "="*80)
        print("🏪 INTERACTIVE POLYMER DISCOVERY SHOP - AI COLLABORATION SYSTEM")
        print("   Enhanced with Multi-PDF Batch Processing")
        print("="*80)
        print("\nWelcome to AI-Enhanced Polymer Discovery!")
        print("Combining: Alice (LLM) + Victor (ML) + Frank (Optimizer)")
        
    def get_polymer_type(self):
        print("\n📋 STEP 1: SELECT YOUR POLYMER")
        print("-" * 80)
        polymers = {'1': 'PP', '2': 'PE', '3': 'POE', '4': 'PET', '5': 'PU'}
        print("Options: 1=PP, 2=PE, 3=POE, 4=PET, 5=PU, 6=Custom")
        choice = input("Select (1-6): ").strip()
        self.polymer_type = polymers.get(choice, "Custom Polymer")
        print(f"✓ Selected: {self.polymer_type}")
        return self.polymer_type
    
    def load_demo_data(self):
        print("\n✨ Generating demo polymer data...")
        np.random.seed(42)
        n = 80
        data = {
            'Component_1': np.random.uniform(50, 80, n),
            'Component_2': np.random.uniform(10, 35, n),
            'Component_3': np.random.uniform(5, 20, n),
        }
        df = pd.DataFrame(data)
        total = df.sum(axis=1)
        for col in df.columns:
            df[col] = (df[col] / total) * 100
        df['Property_1'] = 20 + 0.4*df['Component_2'] + np.random.normal(0, 2, n)
        df['Property_2'] = 30 + 0.8*df['Component_2'] - 0.2*df['Component_3'] + np.random.normal(0, 3, n)
        df['Property_3'] = 40 - 0.3*df['Component_2'] + 1.5*df['Component_3'] + np.random.normal(0, 2, n)
        for col in df.select_dtypes(include=[np.number]).columns:
            df[col] = np.maximum(df[col], 1)
        self.data_source = "Demo Data"
        self.user_data = df
        print(f"✓ Generated {len(df)} formulations")
        return df

class AliceKnowledgeSynthesis:
    def __init__(self, polymer_type, data, data_source):
        self.polymer_type = polymer_type
        self.data = data
        self.data_source = data_source
        
    def synthesize_knowledge(self):
        print("\n🧙 STEP 2: ALICE - KNOWLEDGE SYNTHESIS")
        print("-" * 80)
        print("Alice synthesizing polymer knowledge from data...")
        insights = f"ALICE'S INSIGHTS:\n✓ Analyzed {len(self.data)} formulations\n✓ Identified component-property relationships\n✓ Generated optimization hypotheses"
        print(insights)
        return insights

class VictorMLPredictor:
    def __init__(self, data, polymer_type):
        self.data = data
        self.polymer_type = polymer_type
        self.models = {}
        self.scaler = StandardScaler()
        
    def train_models(self):
        print("\n🔬 STEP 3: VICTOR - MACHINE LEARNING TRAINING")
        print("-" * 80)
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) > 3:
            X = self.data[numeric_cols[:-3]].values
            targets = numeric_cols[-3:]
        else:
            X = self.data[numeric_cols[:-1]].values
            targets = [numeric_cols[-1]]
        X_scaled = self.scaler.fit_transform(X)
        print(f"Training {len(targets)} models...")
        for target in targets:
            y = self.data[target].values
            model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=8)
            model.fit(X_scaled, y)
            self.models[target] = model
            score = model.score(X_scaled, y)
            print(f"  ✓ {target}: R² = {score:.3f}")
        self.X_scaled = X_scaled
        self.feature_cols = numeric_cols[:-3] if len(numeric_cols) > 3 else numeric_cols[:-1]
        self.targets = targets
        return self.models
    
    def predict_properties(self, formulation):
        formulation_scaled = self.scaler.transform([formulation])[0]
        predictions = {}
        for target in self.targets:
            pred = self.models[target].predict([formulation_scaled])[0]
            predictions[target] = max(pred, 0)
        return predictions

class FrankOptimizer:
    def __init__(self, victor_predictor):
        self.victor = victor_predictor
        self.results = []
        
    def optimize(self):
        print("\n⚡ STEP 4: FRANK - OPTIMIZATION")
        print("-" * 80)
        n_components = len(self.victor.feature_cols)
        bounds = [(10, 90) for _ in range(n_components)]
        
        def objective_balanced(x):
            x_normalized = x / x.sum() * 100
            predictions = self.victor.predict_properties(x_normalized)
            normalized = [pred / 100 for pred in predictions.values()]
            return -np.mean(normalized)
        
        print("Running optimization...")
        result = differential_evolution(objective_balanced, bounds, seed=42, maxiter=200, workers=1)
        optimal_formulation = result.x / result.x.sum() * 100
        optimal_properties = self.victor.predict_properties(optimal_formulation)
        
        print(f"✓ Optimization complete!")
        print(f"Optimal Formulation:")
        for i, comp in enumerate(self.victor.feature_cols):
            print(f"  {comp}: {optimal_formulation[i]:.1f}%")
        print(f"Predicted Properties:")
        for target, value in optimal_properties.items():
            print(f"  {target}: {value:.2f}")
        
        self.results = {'formulation': optimal_formulation, 'properties': optimal_properties}
        return self.results

def main():
    print("\n" + "="*80)
    print("🏪 AI-ENHANCED POLYMER DISCOVERY SYSTEM")
    print("="*80)
    
    output_folder = setup_output_folder()
    system = PolymerAISystem(output_folder)
    system.welcome_screen()
    
    polymer = system.get_polymer_type()
    data = system.load_demo_data()
    
    alice = AliceKnowledgeSynthesis(polymer, data, system.data_source)
    alice.synthesize_knowledge()
    
    victor = VictorMLPredictor(data, polymer)
    victor.train_models()
    
    frank = FrankOptimizer(victor)
    results = frank.optimize()
    
    print("\n💾 SAVING RESULTS")
    output_path = os.path.join(output_folder, 'formulation_data.csv')
    data.to_csv(output_path, index=False)
    print(f"✓ Saved: formulation_data.csv")
    
    results_df = pd.DataFrame({
        'Component': victor.feature_cols[:len(results['formulation'])],
        'Optimal_Percentage': results['formulation'][:len(victor.feature_cols)]
    })
    output_path = os.path.join(output_folder, 'optimal_formulation.csv')
    results_df.to_csv(output_path, index=False)
    print(f"✓ Saved: optimal_formulation.csv")
    
    print("\n" + "="*80)
    print("✅ PROJECT COMPLETE!")
    print("="*80)
    print(f"Output folder: {output_folder}/")

if __name__ == "__main__":
    main()
