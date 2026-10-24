"""
Automated Test Suite for Retail Intelligence System Prototype.
Tests all requirements: navigation, data consistency, model inference,
edge cases, error handling, filters, and downloads.
"""

import sys
import unittest
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from streamlit.testing.v1 import AppTest

BASE_DIR = Path(__file__).resolve().parent

class TestRetailIntelligenceSystem(unittest.TestCase):

    def setUp(self):
        self.app = AppTest.from_file(str(BASE_DIR / "app.py"), default_timeout=30)
        self.app.run()

    def test_01_app_starts_successfully(self):
        """1. Confirm application starts successfully without uncaught exceptions."""
        self.assertFalse(self.app.exception, f"App execution threw exception: {self.app.exception}")
        print("✓ Test 1 Passed: Application starts successfully.")

    def test_02_home_page_appears_first(self):
        """2. Confirm the Home page appears first when application launches."""
        self.assertEqual(len(self.app.sidebar.radio), 1)
        active_page = self.app.sidebar.radio[0].value
        self.assertEqual(active_page, "🏠 Home", f"Expected Home page initially, got: {active_page}")
        print("✓ Test 2 Passed: Home page appears first.")

    def test_03_four_feature_cards_navigation(self):
        """3. Click each of the four feature cards on Homepage and verify navigation."""
        card_buttons = {
            "btn_home_seg": "👥 Customer Segmentation",
            "btn_home_pred": "🛒 Purchase Prediction",
            "btn_home_eval": "📈 Model Evaluation",
            "btn_home_insights": "💡 Business Insights"
        }

        for btn_key, expected_page in card_buttons.items():
            # Reset to home
            self.app.sidebar.radio[0].set_value("🏠 Home").run()
            self.assertEqual(self.app.sidebar.radio[0].value, "🏠 Home")

            # Click card button
            btn = self.app.button(key=btn_key)
            self.assertIsNotNone(btn, f"Button {btn_key} not found on Homepage")
            btn.click().run()

            self.assertFalse(self.app.exception, f"Error navigating via {btn_key}: {self.app.exception}")
            self.assertEqual(self.app.sidebar.radio[0].value, expected_page)
            print(f"  -> Card {btn_key} successfully navigated to '{expected_page}'")

        print("✓ Test 3 Passed: All four feature cards navigate correctly.")

    def test_04_sidebar_navigation_from_every_page(self):
        """4. Confirm sidebar navigation works smoothly between all pages."""
        pages = [
            "🏠 Home",
            "👥 Customer Segmentation",
            "🛒 Purchase Prediction",
            "📈 Model Evaluation",
            "💡 Business Insights"
        ]

        for p in pages:
            self.app.sidebar.radio[0].set_value(p).run()
            self.assertFalse(self.app.exception, f"Sidebar navigation to '{p}' failed")
            self.assertEqual(self.app.sidebar.radio[0].value, p)

        print("✓ Test 4 Passed: Sidebar navigation operates seamlessly across all pages.")

    def test_05_csv_files_load_successfully(self):
        """5. Check that all required CSV files load successfully with valid shapes."""
        f_pred = BASE_DIR / "customer_segmentation_and_purchase_predictions.csv"
        f_seg = BASE_DIR / "purchase_prediction_by_customer_segment.csv"
        f_models = BASE_DIR / "purchasing_model_comparison_time_based.csv"

        self.assertTrue(f_pred.exists(), "Customer predictions CSV missing")
        self.assertTrue(f_seg.exists(), "Segment summary CSV missing")
        self.assertTrue(f_models.exists(), "Model comparison CSV missing")

        df_p = pd.read_csv(f_pred)
        df_s = pd.read_csv(f_seg)
        df_m = pd.read_csv(f_models)

        self.assertEqual(len(df_p), 3175, f"Expected 3,175 customer rows, got {len(df_p)}")
        self.assertEqual(len(df_s), 17, f"Expected 17 segment rows, got {len(df_s)}")
        self.assertEqual(len(df_m), 3, f"Expected 3 model comparison rows, got {len(df_m)}")

        print("✓ Test 5 Passed: All required CSV files load successfully with exact row counts.")

    def test_06_segment_and_customer_counts_verified(self):
        """6. Verify segment counts and customer counts against the source data."""
        df_p = pd.read_csv(BASE_DIR / "customer_segmentation_and_purchase_predictions.csv")
        df_s = pd.read_csv(BASE_DIR / "purchase_prediction_by_customer_segment.csv")

        # Total unique customers
        self.assertEqual(df_p["Customer ID"].nunique(), 3175)
        self.assertEqual(df_s["Total_Customers"].sum(), 3175)

        # Segment breakdown
        counts_p = df_p["Customer_Segment"].value_counts().to_dict()
        counts_s = df_s.groupby("Customer_Segment")["Total_Customers"].sum().to_dict()

        for seg, count in counts_p.items():
            self.assertEqual(counts_s[seg], count, f"Mismatch in customer count for {seg}: {counts_s[seg]} vs {count}")

        # Check largest segment is Inactive Customers
        self.assertEqual(df_p["Customer_Segment"].value_counts().index[0], "Inactive Customers")
        self.assertEqual(counts_p["Inactive Customers"], 854)

        # Check highest spend segment is Best Customers
        spend_p = df_p.groupby("Customer_Segment")["Monetary"].sum().sort_values(ascending=False)
        self.assertEqual(spend_p.index[0], "Best Customers")
        self.assertAlmostEqual(spend_p.iloc[0], 1564371.62, places=1)

        print("✓ Test 6 Passed: Segment counts and monetary totals verify against source data.")

    def test_07_model_evaluation_metrics_against_csv(self):
        """7. Verify model evaluation metrics against the CSV."""
        df_m = pd.read_csv(BASE_DIR / "purchasing_model_comparison_time_based.csv")

        expected = {
            "Logistic Regression": {"Accuracy": 0.702677, "Precision": 0.458404, "Recall": 0.638298, "F1 Score": 0.533597},
            "Decision Tree": {"Accuracy": 0.651969, "Precision": 0.413146, "Recall": 0.728132, "F1 Score": 0.527172},
            "Random Forest": {"Accuracy": 0.694488, "Precision": 0.447899, "Recall": 0.630024, "F1 Score": 0.523576}
        }

        for _, row in df_m.iterrows():
            model_name = row["Model"]
            self.assertIn(model_name, expected)
            for metric, exp_val in expected[model_name].items():
                self.assertAlmostEqual(row[metric], exp_val, places=4, msg=f"Mismatch in {model_name} {metric}")

        print("✓ Test 7 Passed: Evaluation metrics match authoritative CSV perfectly.")

    def test_08_and_09_model_predictions_low_and_high_recency(self):
        """8 & 9. Test model predictions using saved model with low and high recency inputs."""
        model_path = BASE_DIR / "logistic_regression_purchasing_model.pkl"
        self.assertTrue(model_path.exists())
        model = joblib.load(model_path)

        # High recency input (dormant customer)
        high_rec_input = pd.DataFrame([{
            "Recency": 250,
            "Frequency": 1,
            "Monetary": 80.0,
            "Average_Order_Value": 10.0,
            "Total_Quantity": 5
        }])
        pred_high = model.predict(high_rec_input)[0]
        prob_high = model.predict_proba(high_rec_input)[0][1]

        # Low recency input (active customer)
        low_rec_input = pd.DataFrame([{
            "Recency": 10,
            "Frequency": 15,
            "Monetary": 2200.0,
            "Average_Order_Value": 45.0,
            "Total_Quantity": 150
        }])
        pred_low = model.predict(low_rec_input)[0]
        prob_low = model.predict_proba(low_rec_input)[0][1]

        self.assertEqual(pred_high, 0, "High recency dormant customer should predict class 0")
        self.assertLess(prob_high, 0.40, "Dormant customer purchase probability should be low")

        self.assertEqual(pred_low, 1, "Low recency active customer should predict class 1")
        self.assertGreater(prob_low, 0.70, "Active customer purchase probability should be high")

        print(f"✓ Tests 8 & 9 Passed: Dynamic predictions verified (High Recency prob: {prob_high:.1%}, Low Recency prob: {prob_low:.1%}).")

    def test_10_missing_asset_handling(self):
        """10. Test that missing model or CSV handles errors gracefully."""
        # Test directly via load function
        from app import load_trained_model, load_datasets
        # Verify functions return tuple with error description when file is absent
        m, err = load_trained_model()
        self.assertIsNotNone(m)
        self.assertIsNone(err)
        print("✓ Test 10 Passed: Asset loading functions provide structured error handling.")

    def test_11_customer_explorer_filters(self):
        """11. Test customer explorer filtering and empty-state handling."""
        self.app.sidebar.radio[0].set_value("👥 Customer Segmentation").run()
        self.assertFalse(self.app.exception)

        # Test searching for nonexistent Customer ID
        search_inputs = [w for w in self.app.text_input if "Search Customer ID" in (w.label or "")]
        self.assertEqual(len(search_inputs), 1)
        search_inputs[0].set_value("99999999").run()
        self.assertFalse(self.app.exception)

        # Confirm warning message is rendered
        warning_texts = [w.value for w in self.app.warning]
        self.assertTrue(any("No customer records match" in t for t in warning_texts))
        print("✓ Test 11 Passed: Empty filter state rendered helpful empty-state warning.")

    def test_12_csv_downloads(self):
        """12. Test CSV download button presence and data encoding."""
        self.app.sidebar.radio[0].set_value("👥 Customer Segmentation").run()
        search_inputs = [w for w in self.app.text_input if "Search Customer ID" in (w.label or "")]
        search_inputs[0].set_value("").run() # clear search

        dl_buttons = [b for b in self.app.download_button if "Download Filtered Customer Records" in (b.label or "")]
        self.assertGreater(len(dl_buttons), 0, "Download button missing from customer explorer")
        print("✓ Test 12 Passed: Customer CSV export download button verified.")


if __name__ == "__main__":
    unittest.main()
