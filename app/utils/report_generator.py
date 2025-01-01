import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import jinja2
import pdfkit
import json
from pathlib import Path
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from app.utils.visualizations import AdvancedVisualizations
from app.utils.analytics import PortfolioAnalytics, TradeAnalytics, RiskAnalytics

class ReportGenerator:
    def __init__(self, template_dir: str):
        self.template_loader = jinja2.FileSystemLoader(searchpath=template_dir)
        self.template_env = jinja2.Environment(loader=self.template_loader)
        self.visualizations = AdvancedVisualizations()
        self.portfolio_analytics = PortfolioAnalytics()
        self.trade_analytics = TradeAnalytics()
        self.risk_analytics = RiskAnalytics()

    def generate_daily_report(self, portfolio_data: Dict, trades_data: Dict,
                            output_dir: str) -> str:
        """Generate daily performance report"""
        template = self.template_env.get_template('daily_report.html')
        
        # Calculate metrics
        daily_metrics = self._calculate_daily_metrics(portfolio_data, trades_data)
        
        # Generate visualizations
        charts = self._generate_daily_charts(portfolio_data, trades_data)
        
        # Render template
        html_content = template.render(
            date=datetime.now().strftime('%Y-%m-%d'),
            metrics=daily_metrics,
            charts=charts
        )
        
        # Convert to PDF
        output_path = Path(output_dir) / f"daily_report_{datetime.now().strftime('%Y%m%d')}.pdf"
        pdfkit.from_string(html_content, str(output_path))
        
        return str(output_path)

    def generate_weekly_report(self, portfolio_data: Dict, trades_data: Dict,
                             output_dir: str) -> str:
        """Generate weekly performance report"""
        template = self.template_env.get_template('weekly_report.html')
        
        # Calculate weekly metrics
        weekly_metrics = self._calculate_weekly_metrics(portfolio_data, trades_data)
        
        # Generate visualizations
        charts = self._generate_weekly_charts(portfolio_data, trades_data)
        
        # Render template
        html_content = template.render(
            week_ending=datetime.now().strftime('%Y-%m-%d'),
            metrics=weekly_metrics,
            charts=charts
        )
        
        # Convert to PDF
        output_path = Path(output_dir) / f"weekly_report_{datetime.now().strftime('%Y%m%d')}.pdf"
        pdfkit.from_string(html_content, str(output_path))
        
        return str(output_path)

    def generate_monthly_report(self, portfolio_data: Dict, trades_data: Dict,
                              output_dir: str) -> str:
        """Generate monthly performance report"""
        template = self.template_env.get_template('monthly_report.html')
        
        # Calculate monthly metrics
        monthly_metrics = self._calculate_monthly_metrics(portfolio_data, trades_data)
        
        # Generate visualizations
        charts = self._generate_monthly_charts(portfolio_data, trades_data)
        
        # Render template
        html_content = template.render(
            month=datetime.now().strftime('%B %Y'),
            metrics=monthly_metrics,
            charts=charts
        )
        
        # Convert to PDF
        output_path = Path(output_dir) / f"monthly_report_{datetime.now().strftime('%Y%m')}.pdf"
        pdfkit.from_string(html_content, str(output_path))
        
        return str(output_path)

    def send_report_email(self, report_path: str, recipient_email: str,
                         email_config: Dict) -> None:
        """Send report via email"""
        msg = MIMEMultipart()
        msg['Subject'] = f"Trading Performance Report - {datetime.now().strftime('%Y-%m-%d')}"
        msg['From'] = email_config['sender_email']
        msg['To'] = recipient_email
        
        # Add body
        body = "Please find attached your trading performance report."
        msg.attach(MIMEText(body, 'plain'))
        
        # Add PDF attachment
        with open(report_path, 'rb') as f:
            pdf = MIMEApplication(f.read(), _subtype='pdf')
            pdf.add_header('Content-Disposition', 'attachment',
                          filename=Path(report_path).name)
            msg.attach(pdf)
        
        # Send email
        with smtplib.SMTP_SSL(email_config['smtp_server'],
                             email_config['smtp_port']) as server:
            server.login(email_config['username'], email_config['password'])
            server.send_message(msg)

    def _calculate_daily_metrics(self, portfolio_data: Dict,
                               trades_data: Dict) -> Dict:
        """Calculate daily performance metrics"""
        return {
            'portfolio_value': portfolio_data['current_value'],
            'daily_return': portfolio_data['daily_return'],
            'daily_pnl': portfolio_data['daily_pnl'],
            'trades_count': len(trades_data['daily_trades']),
            'win_rate': trades_data['daily_win_rate'],
            'largest_win': trades_data['daily_largest_win'],
            'largest_loss': trades_data['daily_largest_loss'],
            'sharpe_ratio': portfolio_data['daily_sharpe'],
            'max_drawdown': portfolio_data['daily_drawdown']
        }

    def _calculate_weekly_metrics(self, portfolio_data: Dict,
                                trades_data: Dict) -> Dict:
        """Calculate weekly performance metrics"""
        return {
            'weekly_return': portfolio_data['weekly_return'],
            'weekly_pnl': portfolio_data['weekly_pnl'],
            'trades_count': len(trades_data['weekly_trades']),
            'win_rate': trades_data['weekly_win_rate'],
            'best_day': portfolio_data['weekly_best_day'],
            'worst_day': portfolio_data['weekly_worst_day'],
            'sharpe_ratio': portfolio_data['weekly_sharpe'],
            'sortino_ratio': portfolio_data['weekly_sortino'],
            'max_drawdown': portfolio_data['weekly_drawdown']
        }

    def _calculate_monthly_metrics(self, portfolio_data: Dict,
                                 trades_data: Dict) -> Dict:
        """Calculate monthly performance metrics"""
        return {
            'monthly_return': portfolio_data['monthly_return'],
            'monthly_pnl': portfolio_data['monthly_pnl'],
            'trades_count': len(trades_data['monthly_trades']),
            'win_rate': trades_data['monthly_win_rate'],
            'best_day': portfolio_data['monthly_best_day'],
            'worst_day': portfolio_data['monthly_worst_day'],
            'sharpe_ratio': portfolio_data['monthly_sharpe'],
            'sortino_ratio': portfolio_data['monthly_sortino'],
            'max_drawdown': portfolio_data['monthly_drawdown'],
            'alpha': portfolio_data['monthly_alpha'],
            'beta': portfolio_data['monthly_beta']
        }

    def _generate_daily_charts(self, portfolio_data: Dict,
                             trades_data: Dict) -> Dict:
        """Generate charts for daily report"""
        return {
            'intraday_pnl': self.visualizations.create_pnl_distribution(
                pd.Series(portfolio_data['intraday_returns'])
            ),
            'hourly_volume': self.visualizations.create_returns_heatmap(
                pd.DataFrame(trades_data['hourly_volume'])
            )
        }

    def _generate_weekly_charts(self, portfolio_data: Dict,
                              trades_data: Dict) -> Dict:
        """Generate charts for weekly report"""
        return {
            'daily_returns': self.visualizations.create_rolling_metrics(
                pd.Series(portfolio_data['daily_returns']), window=5
            ),
            'asset_correlation': self.visualizations.create_correlation_heatmap(
                pd.DataFrame(portfolio_data['asset_returns'])
            ),
            'drawdown': self.visualizations.create_drawdown_chart(
                pd.Series(portfolio_data['equity_curve'])
            )
        }

    def _generate_monthly_charts(self, portfolio_data: Dict,
                               trades_data: Dict) -> Dict:
        """Generate charts for monthly report"""
        return {
            'monthly_performance': self.visualizations.create_rolling_metrics(
                pd.Series(portfolio_data['daily_returns']), window=21
            ),
            'risk_metrics': self.visualizations.create_risk_metrics_radar(
                portfolio_data['risk_metrics']
            ),
            'factor_exposure': self.visualizations.create_factor_exposure_heatmap(
                pd.DataFrame(portfolio_data['factor_exposures'])
            )
        }
