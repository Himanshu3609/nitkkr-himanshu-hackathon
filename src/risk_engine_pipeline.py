import pandas as pd
import numpy as np
from transformers import pipeline
import warnings

warnings.filterwarnings('ignore')

class RiskEnginePipeline:
    """
    Unified AI/NLP Pipeline that orchestrates the flow of data through models.
    """
    def __init__(self):
        print("Initializing Advanced NLP Models...")
        # Using industry-standard FinBERT for financial sentiment
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis", 
            model="ProsusAI/finbert"
        )
        
        # Using DistilBERT for zero-shot: MUCH faster and smaller than BART-Large
        self.event_classifier = pipeline(
            "zero-shot-classification", 
            model="typeform/distilbert-base-uncased-mnli"
        )
        
        self.event_labels = [
            "Geopolitical Event", 
            "Macroeconomic Shift", 
            "Credit Risk", 
            "Merger and Acquisition", 
            "Product Launch",
            "Earnings Report",
            "Regulatory Action",
            "Executive Leadership Change",
            "Legal & Litigation",
            "Supply Chain Disruption",
            "Cybersecurity & Data Breach",
            "Dividend & Share Buyback",
            "Bankruptcy & Restructuring",
            "ESG & Environmental Risk"
        ]
        print("Pipeline initialized successfully.")

    def run_pipeline(self, text, source="Unknown"):
        """
        Executes the end-to-end processing pipeline for a single text input.
        """
        if not text or not isinstance(text, str):
            return self._default_response()

        # Step 1: Sentiment Extraction
        sentiment_score = self._extract_sentiment(text)
        
        # Step 2: Event Classification
        event_type, confidence = self._classify_event(text)
        
        # Step 3: Impact Scoring
        impact_score = self._calculate_impact(sentiment_score, confidence, event_type)

        return {
            "source": source,
            "text": text,
            "sentiment_score": sentiment_score,
            "event_class": event_type,
            "impact_score": impact_score,
            "confidence": confidence
        }

    def _extract_sentiment(self, text):
        try:
            # FinBERT outputs positive, negative, neutral
            sent_result = self.sentiment_analyzer(text[:512])[0] 
            label = sent_result['label'].lower()
            score = sent_result['score']
            
            if label == 'positive':
                return score
            elif label == 'negative':
                return -score
            else: # neutral
                # Return a small score close to 0 but retaining some signal
                return score * 0.1 
        except Exception as e:
            return 0.0

    def _classify_event(self, text):
        try:
            class_result = self.event_classifier(text[:512], candidate_labels=self.event_labels)
            return class_result['labels'][0], class_result['scores'][0]
        except Exception as e:
            return "Unknown", 0.0

    def _calculate_impact(self, sentiment_score, confidence, event_type):
        # Base impact from sentiment magnitude and classification confidence
        base_impact = (abs(sentiment_score) * 6) + (confidence * 4)
        
        # Modifiers based on event type severity - expanded to include many critical risk categories
        high_severity = [
            "Geopolitical Event", 
            "Macroeconomic Shift", 
            "Credit Risk",
            "Bankruptcy & Restructuring",
            "Regulatory Action",
            "Cybersecurity & Data Breach",
            "Supply Chain Disruption",
            "Legal & Litigation"
        ]
        if event_type in high_severity:
            base_impact += 2.5
            
        impact_score = min(max(int(round(base_impact)), 1), 10)
        return impact_score

    def _default_response(self):
        return {
            "source": "Unknown",
            "text": "",
            "sentiment_score": 0.0,
            "event_class": "None",
            "impact_score": 1,
            "confidence": 0.0
        }
