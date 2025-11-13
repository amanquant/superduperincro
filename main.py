"""
Incrolink Finance Analysis Platform
Non-Streamlit version for local deployment and GitHub
Core finance engine with Flask API and CLI interfaces
"""

import pandas as pd
import numpy as np
import sys
import json
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, asdict
import logging

# Import core finance functions
from finance_core import (
    validate_columns, extract_date_columns, extract_financial_statement_data,
    calculate_ratios_from_financial_statement, calculate_metrics_from_dataset,
    get_company_category_code, get_sector_percentiles, get_percentile_position,
    get_ceo_age, get_contacts_by_company_id, get_contact_by_id,
    get_related_contacts_by_relative, predictability_decision_tree,
    fuzzy_match_companies, DCF_automated,
    COLUMNS_REQUIRED, COLUMNS_PORTFOLIO, FINANCIAL_ITEMS,
    PREDICTABILITY_CATEGORIES
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class CompanyAnalysis:
    """Company financial analysis result"""
    company_name: str
    category_code: str
    metrics: Dict
    sector_percentiles: Dict
    dcf_result: Dict
    predictability: Dict

    def to_json(self) -> str:
        """Convert to JSON"""
        return json.dumps(asdict(self), indent=2, default=str)


@dataclass
class DealMatch:
    """Deal matching result"""
    portfolio_company: str
    target_company: str
    category_code: str
    buyer_fit: str
    classification: str  # "Top Pick", "Good Deal", "Bit Overvalued"
    growth_expected: float
    dcf_result: Dict


# ============================================================================
# ANALYSIS ENGINE
# ============================================================================

class FinanceEngine:
    """Core finance analysis engine"""

    def __init__(self, dataset_path: str, waccmap_path: str, contacts_path: Optional[str] = None):
        """Initialize with data files"""
        logger.info("Initializing Finance Engine...")

        self.dataset = pd.read_excel(dataset_path)
        self.waccmap = pd.read_excel(waccmap_path)
        self.contacts = pd.read_excel(contacts_path) if contacts_path else None

        # Validate dataset
        is_valid, msg = validate_columns(self.dataset)
        if not is_valid:
            logger.error(msg)
            raise ValueError(msg)

        logger.info(f"✅ Loaded {len(self.dataset)} companies, {len(self.waccmap)} WACC categories")

    def analyze_company(self, company_name: str, financial_statement_path: Optional[str] = None) -> CompanyAnalysis:
        """Analyze a single company"""
        logger.info(f"Analyzing company: {company_name}")

        # Find company in dataset
        matching = self.dataset[self.dataset['company'].str.contains(company_name, case=False, na=False)]

        if matching.empty:
            raise ValueError(f"Company '{company_name}' not found")

        company_row = matching.iloc[0]

        # Extract metrics from financial statement or dataset
        metrics = {}
        if financial_statement_path:
            fs_df = pd.read_excel(financial_statement_path)
            date_cols = extract_date_columns(fs_df)
            items_found, _ = extract_financial_statement_data(fs_df, date_cols)
            metrics = calculate_ratios_from_financial_statement(items_found)
        else:
            metrics = calculate_metrics_from_dataset(company_row)

        # Get sector data
        category_code = get_company_category_code(company_name, self.dataset)
        sector_percentiles = get_sector_percentiles(category_code, self.waccmap)

        # Calculate DCF
        dcf_result = DCF_automated(company_row, self.waccmap)

        # Predictability analysis
        ev_growth = dcf_result['growth_expected']
        revenue = company_row.get('net income', np.nan)
        edamargin = metrics.get('edamargin', np.nan)
        edamargin_p75 = sector_percentiles.get('edamargin', {}).get('p75', np.nan)
        nsellside = sector_percentiles.get('nsellside', np.nan)
        nsellside_p50 = sector_percentiles.get('nsellside_p50', np.nan)
        ceo_age = get_ceo_age(company_row, self.contacts)

        leaf_value, category, path = predictability_decision_tree(
            ev_growth, nsellside, nsellside_p50, ceo_age, revenue, edamargin, edamargin_p75
        )

        predictability = {
            'leaf_value': leaf_value,
            'category': category,
            'decision_path': path
        }

        return CompanyAnalysis(
            company_name=company_name,
            category_code=category_code,
            metrics=metrics,
            sector_percentiles=sector_percentiles,
            dcf_result=dcf_result,
            predictability=predictability
        )

    def find_deals(self, portfolio_path: str, investment_style: str = "Good Deals") -> List[DealMatch]:
        """Find deal matches between portfolio and dataset"""
        logger.info(f"Finding deals with style: {investment_style}")

        portfolio_df = pd.read_excel(portfolio_path)
        is_valid, msg = validate_columns(portfolio_df, "Portfolio", COLUMNS_PORTFOLIO)

        if not is_valid:
            logger.error(msg)
            raise ValueError(msg)

        deals = []

        # Calculate metrics for all companies in dataset
        db_metrics_list = []
        for idx, row in self.dataset.iterrows():
            metrics = calculate_metrics_from_dataset(row)
            db_metrics_list.append(metrics)

        metrics_df = pd.DataFrame(db_metrics_list)
        self.dataset['ltde'] = metrics_df['ltde']
        self.dataset['fx'] = metrics_df['fx']
        self.dataset['edamargin'] = metrics_df['edamargin']

        # Weighted score
        self.dataset['weighted_score'] = (
            0.5 * self.dataset['ltde'].fillna(0) +
            0.35 * self.dataset['fx'].fillna(0) +
            0.15 * self.dataset['edamargin'].fillna(0)
        )

        # Get 90th percentile companies
        percentile_90 = self.dataset['weighted_score'].quantile(0.90)
        list_b = self.dataset[self.dataset['weighted_score'] >= percentile_90].copy()

        logger.info(f"Found {len(list_b)} companies at 90th percentile")

        # Match with portfolio
        for idx_a, company_a in portfolio_df.iterrows():
            category_a = company_a.get('category_code', None)

            if category_a is None:
                continue

            matched_b = list_b[list_b['category_code'] == category_a]

            for idx_b, company_b in matched_b.iterrows():
                dcf_result = DCF_automated(company_b, self.waccmap)
                growth_expected = dcf_result['growth_expected']

                # Classify
                if growth_expected < 0:
                    classification = "Bit Overvalued"
                elif 0 <= growth_expected < 0.20:
                    classification = "Good Deal"
                else:
                    classification = "Top Pick"

                # Filter by investment style
                if investment_style == "Top Picks" and classification != "Top Pick":
                    continue
                elif investment_style == "Good Deals" and classification != "Good Deal":
                    continue

                ebit_a = company_a.get('ebit', 0)
                ebit_b = company_b.get('ebit', 0)
                buyer_fit = "Buyer fit: " + company_a['company'] if ebit_a >= ebit_b else "Buyer fit: " + company_b['company']

                deal = DealMatch(
                    portfolio_company=company_a['company'],
                    target_company=company_b['company'],
                    category_code=category_a,
                    buyer_fit=buyer_fit,
                    classification=classification,
                    growth_expected=growth_expected,
                    dcf_result=dcf_result
                )

                deals.append(deal)

        logger.info(f"Found {len(deals)} deal matches")
        return deals

    def search_companies(self, query: str) -> pd.DataFrame:
        """Search companies by name"""
        results = self.dataset[self.dataset['company'].str.contains(query, case=False, na=False)]
        return results


# ============================================================================
# CLI INTERFACE
# ============================================================================

def cli_main():
    """Command-line interface"""
    import argparse

    parser = argparse.ArgumentParser(description='Incrolink Finance Analysis CLI')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze a company')
    analyze_parser.add_argument('company_name', help='Company name')
    analyze_parser.add_argument('--dataset', required=True, help='Path to dataset XLSX')
    analyze_parser.add_argument('--waccmap', required=True, help='Path to WACC map XLSX')
    analyze_parser.add_argument('--contacts', help='Path to contacts XLSX (optional)')
    analyze_parser.add_argument('--fs', help='Financial statement XLSX (optional)')
    analyze_parser.add_argument('--output', help='Output JSON file')

    # Search command
    search_parser = subparsers.add_parser('search', help='Search companies')
    search_parser.add_argument('query', help='Search query')
    search_parser.add_argument('--dataset', required=True, help='Path to dataset XLSX')
    search_parser.add_argument('--waccmap', required=True, help='Path to WACC map XLSX')

    # Deals command
    deals_parser = subparsers.add_parser('deals', help='Find deal matches')
    deals_parser.add_argument('portfolio', help='Path to portfolio XLSX')
    deals_parser.add_argument('--dataset', required=True, help='Path to dataset XLSX')
    deals_parser.add_argument('--waccmap', required=True, help='Path to WACC map XLSX')
    deals_parser.add_argument('--style', default='Good Deals', help='Investment style')
    deals_parser.add_argument('--output', help='Output JSON file')

    args = parser.parse_args()

    if args.command == 'analyze':
        engine = FinanceEngine(args.dataset, args.waccmap, args.contacts)
        analysis = engine.analyze_company(args.company_name, args.fs)

        output = analysis.to_json()
        print(output)

        if args.output:
            with open(args.output, 'w') as f:
                f.write(output)
            logger.info(f"Results saved to {args.output}")

    elif args.command == 'search':
        engine = FinanceEngine(args.dataset, args.waccmap)
        results = engine.search_companies(args.query)
        print(results.to_string())

    elif args.command == 'deals':
        engine = FinanceEngine(args.dataset, args.waccmap)
        deals = engine.find_deals(args.portfolio, args.style)

        output = json.dumps([asdict(d) for d in deals], indent=2, default=str)
        print(output)

        if args.output:
            with open(args.output, 'w') as f:
                f.write(output)
            logger.info(f"Results saved to {args.output}")

    else:
        parser.print_help()


# ============================================================================
# FLASK API INTERFACE (Optional)
# ============================================================================

try:
    from flask import Flask, request, jsonify, render_template

    app = Flask(__name__)
    engine = None  # Will be initialized on startup

    @app.route('/api/analyze', methods=['POST'])
    def api_analyze():
        """Analyze company via API"""
        try:
            data = request.json
            company_name = data.get('company_name')

            if not company_name:
                return jsonify({'error': 'company_name required'}), 400

            analysis = engine.analyze_company(company_name)
            return jsonify(asdict(analysis)), 200

        except Exception as e:
            return jsonify({'error': str(e)}), 400

    @app.route('/api/search', methods=['GET'])
    def api_search():
        """Search companies via API"""
        query = request.args.get('q', '')
        results = engine.search_companies(query)
        return jsonify(results.to_dict(orient='records')), 200

    @app.route('/api/deals', methods=['POST'])
    def api_deals():
        """Find deals via API"""
        try:
            # Expects portfolio data in request
            # Implementation depends on file upload handling
            return jsonify({'message': 'Not implemented'}), 501
        except Exception as e:
            return jsonify({'error': str(e)}), 400

    @app.route('/', methods=['GET'])
    def index():
        """Root endpoint"""
        return jsonify({
            'name': 'Incrolink Finance API',
            'version': '1.0',
            'endpoints': [
                '/api/analyze',
                '/api/search',
                '/api/deals'
            ]
        }), 200

    def init_engine(dataset_path: str, waccmap_path: str, contacts_path: Optional[str] = None):
        """Initialize Flask app with engine"""
        global engine
        engine = FinanceEngine(dataset_path, waccmap_path, contacts_path)

except ImportError:
    logger.warning("Flask not installed - API interface disabled")
    app = None


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    if len(sys.argv) > 1:
        cli_main()
    else:
        print("Incrolink Finance Analysis Platform")
        print("\nUsage:")
        print("  python main.py analyze <company> --dataset <path> --waccmap <path>")
        print("  python main.py search <query> --dataset <path> --waccmap <path>")
        print("  python main.py deals <portfolio> --dataset <path> --waccmap <path>")
        print("\nFor Flask API:")
        print("  from main import app, init_engine")
        print("  init_engine('dataset.xlsx', 'wacc.xlsx')")
        print("  app.run(debug=True)")
