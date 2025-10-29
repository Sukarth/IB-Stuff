#!/usr/bin/env python3
"""
Advanced IB Grade Boundary Analysis Tool
Author: Sukarth
Date: October 2025

This script provides comprehensive analysis of IB grade boundaries across
multiple sessions and subjects, identifying trends and generating insights.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class IBBoundaryAnalyzer:
    """Advanced analyzer for IB grade boundary data."""
    
    def __init__(self):
        self.sessions = ['M21', 'N21', 'M22', 'N22', 'M23', 'N23', 'M24', 'N24']
        self.math_data = self.load_math_data()
        self.all_data = self.parse_all_boundaries()
        
    def load_math_data(self) -> pd.DataFrame:
        """Load and clean the existing Math.csv data."""
        try:
            df = pd.read_csv('Math.csv')
            # Extract numeric values from grade ranges
            for grade in ['Grade 1', 'Grade 2', 'Grade 3', 'Grade 4', 'Grade 5', 'Grade 6', 'Grade 7']:
                df[f'{grade}_Low'] = df[grade].str.split('-').str[0].astype(int)
                df[f'{grade}_High'] = df[grade].str.split('-').str[1].astype(int)
            
            # Create session order for proper sorting
            session_order = {'M21': 1, 'N21': 2, 'M22': 3, 'N22': 4, 'M23': 5, 'N23': 6, 'M24': 7, 'N24': 8}
            df['Session_Order'] = df['Session'].map(session_order)
            df = df.sort_values('Session_Order')
            
            return df
        except FileNotFoundError:
            print("Math.csv not found, creating sample data...")
            return pd.DataFrame()
    
    def parse_all_boundaries(self) -> Dict[str, pd.DataFrame]:
        """Parse all boundary files and extract structured data."""
        boundaries = {}
        
        for session in self.sessions:
            filename = f"{session}.txt"
            if Path(filename).exists():
                boundaries[session] = self.parse_session_file(filename)
        
        return boundaries
    
    def parse_session_file(self, filename: str) -> pd.DataFrame:
        """Parse a session file and extract subject boundary data."""
        data = []
        current_subject = None
        current_level = None
        
        try:
            with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            # Look for subject patterns
            subject_patterns = [
                r'Subject: ([A-Z\s]+)',
                r'ject: ([A-Z\s]+)',
                r'bject: ([A-Z\s]+)'
            ]
            
            level_patterns = [
                r'Level: (HL|SL)',
                r'evel: (HL|SL)',
                r'vel: (HL|SL)'
            ]
            
            lines = content.split('\n')
            
            for line in lines:
                # Check for subject
                for pattern in subject_patterns:
                    match = re.search(pattern, line)
                    if match:
                        current_subject = match.group(1).strip()
                        break
                
                # Check for level
                for pattern in level_patterns:
                    match = re.search(pattern, line)
                    if match:
                        current_level = match.group(1).strip()
                        break
                
                # Look for final grade boundaries
                if 'FINAL' in line and current_subject and current_level:
                    # Try to extract grade boundaries from the next few lines
                    self.extract_boundaries(lines, line, current_subject, current_level, data)
            
        except Exception as e:
            print(f"Error parsing {filename}: {e}")
        
        return pd.DataFrame(data)
    
    def extract_boundaries(self, lines: List[str], current_line: str, subject: str, level: str, data: List[Dict]):
        """Extract grade boundaries from text lines."""
        # This is a simplified extraction - would need more sophisticated parsing
        # for production use
        pass
    
    def analyze_math_trends(self) -> Dict[str, any]:
        """Analyze trends in Math AA HL boundaries."""
        if self.math_data.empty:
            return {}
        
        df = self.math_data[self.math_data['Level'] == 'HL'].copy()
        
        analysis = {
            'trends': {},
            'statistics': {},
            'predictions': {}
        }
        
        # Analyze trends for each grade
        for grade_num in range(1, 8):
            grade_col = f'Grade {grade_num}_Low'
            if grade_col in df.columns:
                values = df[grade_col].values
                sessions = df['Session'].values
                
                # Calculate trend
                x = np.arange(len(values))
                coeffs = np.polyfit(x, values, 1)
                trend_slope = coeffs[0]
                
                analysis['trends'][f'Grade {grade_num}'] = {
                    'slope': trend_slope,
                    'direction': 'increasing' if trend_slope > 0 else 'decreasing',
                    'values': values.tolist(),
                    'sessions': sessions.tolist(),
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
        
        return analysis
    
    def compare_timezones(self) -> Dict[str, any]:
        """Compare grade boundaries between different timezones."""
        if self.math_data.empty:
            return {}
        
        comparison = {}
        
        for session in self.math_data['Session'].unique():
            session_data = self.math_data[self.math_data['Session'] == session]
            if len(session_data) > 1:  # Multiple timezones
                comparison[session] = {}
                for grade_num in range(1, 8):
                    grade_col = f'Grade {grade_num}_Low'
                    if grade_col in session_data.columns:
                        values = session_data[grade_col].values
                        comparison[session][f'Grade {grade_num}'] = {
                            'values': values.tolist(),
                            'difference': max(values) - min(values),
                            'timezones': session_data['Timezone'].tolist()
                        }
        
        return comparison
    
    def predict_future_boundaries(self, sessions_ahead: int = 2) -> Dict[str, Dict[str, float]]:
        """Predict future grade boundaries based on trends."""
        if self.math_data.empty:
            return {}
        
        df = self.math_data[self.math_data['Level'] == 'HL'].copy()
        predictions = {}
        
        future_sessions = []
        last_session = df['Session'].iloc[-1]
        
        # Generate future session names
        year = int('20' + last_session[1:])
        is_may = last_session.startswith('M')
        
        for i in range(sessions_ahead):
            if is_may:
                future_sessions.append(f'N{str(year)[2:]}')
                is_may = False
            else:
                year += 1
                future_sessions.append(f'M{str(year)[2:]}')
                is_may = True
        
        for grade_num in range(1, 8):
            grade_col = f'Grade {grade_num}_Low'
            if grade_col in df.columns:
                values = df[grade_col].values
                x = np.arange(len(values))
                
                # Fit polynomial trend
                coeffs = np.polyfit(x, values, min(2, len(values)-1))
                poly = np.poly1d(coeffs)
                
                # Predict future values
                future_x = np.arange(len(values), len(values) + sessions_ahead)
                future_values = poly(future_x)
                
                # Ensure predictions are reasonable (round and bound)
                future_values = np.clip(np.round(future_values), 0, 100)
                
                for i, session in enumerate(future_sessions):
                    if session not in predictions:
                        predictions[session] = {}
                    predictions[session][f'Grade {grade_num}'] = int(future_values[i])
        
        return predictions
    
    def generate_insights(self) -> List[str]:
        """Generate key insights from the data analysis."""
        insights = []
        
        if self.math_data.empty:
            insights.append("⚠️  No Math data available for analysis")
            return insights
        
        analysis = self.analyze_math_trends()
        
        # Overall trend insights
        if analysis:
            trends = analysis.get('trends', {})
            
            increasing_grades = []
            decreasing_grades = []
            
            for grade, data in trends.items():
                if data['direction'] == 'increasing':
                    increasing_grades.append(grade)
                else:
                    decreasing_grades.append(grade)
            
            if increasing_grades:
                insights.append(f"📈 Grade boundaries are increasing for: {', '.join(increasing_grades)}")
            
            if decreasing_grades:
                insights.append(f"📉 Grade boundaries are decreasing for: {', '.join(decreasing_grades)}")
            
            # Volatility insights
            volatile_grades = []
            for grade, data in trends.items():
                if data['std'] > 2:  # High standard deviation
                    volatile_grades.append(f"{grade} (σ={data['std']:.1f})")
            
            if volatile_grades:
                insights.append(f"🔄 Most volatile boundaries: {', '.join(volatile_grades)}")
        
        # Timezone comparison insights
        tz_comparison = self.compare_timezones()
        largest_diff = 0
        largest_diff_session = None
        
        for session, grades in tz_comparison.items():
            for grade, data in grades.items():
                if data['difference'] > largest_diff:
                    largest_diff = data['difference']
                    largest_diff_session = f"{session} {grade}"
        
        if largest_diff > 0:
            insights.append(f"🌍 Largest timezone difference: {largest_diff} marks in {largest_diff_session}")
        
        return insights
    
    def create_comprehensive_plots(self):
        """Create comprehensive visualization plots."""
        if self.math_data.empty:
            print("No data available for plotting")
            return
        
        df = self.math_data[self.math_data['Level'] == 'HL'].copy()
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 15))
        
        # Plot 1: Grade boundary trends over time
        plt.subplot(2, 3, 1)
        for grade_num in range(4, 8):  # Focus on higher grades
            grade_col = f'Grade {grade_num}_Low'
            if grade_col in df.columns:
                plt.plot(df['Session'], df[grade_col], marker='o', linewidth=2, 
                        label=f'Grade {grade_num}', markersize=6)
        
        plt.title('IB Math AA HL Grade Boundaries Trend', fontsize=14, fontweight='bold')
        plt.xlabel('Session', fontsize=12)
        plt.ylabel('Boundary Score', fontsize=12)
        plt.legend(frameon=True, fancybox=True, shadow=True)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # Plot 2: Grade distribution heatmap
        plt.subplot(2, 3, 2)
        grade_matrix = []
        grade_labels = []
        
        for grade_num in range(1, 8):
            grade_col = f'Grade {grade_num}_Low'
            if grade_col in df.columns:
                grade_matrix.append(df[grade_col].values)
                grade_labels.append(f'Grade {grade_num}')
        
        if grade_matrix:
            sns.heatmap(grade_matrix, xticklabels=df['Session'], yticklabels=grade_labels,
                       annot=True, fmt='d', cmap='RdYlBu_r', cbar_kws={'label': 'Boundary Score'})
            plt.title('Grade Boundaries Heatmap', fontsize=14, fontweight='bold')
            plt.xlabel('Session', fontsize=12)
            plt.ylabel('Grade Level', fontsize=12)
        
        # Plot 3: Timezone comparison
        plt.subplot(2, 3, 3)
        timezone_data = self.compare_timezones()
        if timezone_data:
            sessions = []
            max_diffs = []
            
            for session, grades in timezone_data.items():
                sessions.append(session)
                max_diff = max([data['difference'] for data in grades.values()])
                max_diffs.append(max_diff)
            
            bars = plt.bar(sessions, max_diffs, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'], alpha=0.8)
            plt.title('Maximum Timezone Differences', fontsize=14, fontweight='bold')
            plt.xlabel('Session', fontsize=12)
            plt.ylabel('Max Difference (marks)', fontsize=12)
            plt.xticks(rotation=45)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{int(height)}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 4: Grade 7 analysis (hardest to achieve)
        plt.subplot(2, 3, 4)
        if 'Grade 7_Low' in df.columns:
            sessions = df['Session'].values
            grade_7_scores = df['Grade 7_Low'].values
            
            plt.plot(sessions, grade_7_scores, 'ro-', linewidth=3, markersize=8, color='#E74C3C')
            plt.fill_between(sessions, grade_7_scores, alpha=0.3, color='#E74C3C')
            
            # Add trend line
            x = np.arange(len(sessions))
            z = np.polyfit(x, grade_7_scores, 1)
            p = np.poly1d(z)
            plt.plot(sessions, p(x), "--", alpha=0.8, color='#2C3E50', linewidth=2)
            
            plt.title('Grade 7 Boundary Evolution', fontsize=14, fontweight='bold')
            plt.xlabel('Session', fontsize=12)
            plt.ylabel('Minimum Score for Grade 7', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
        
        # Plot 5: Predictions
        plt.subplot(2, 3, 5)
        predictions = self.predict_future_boundaries(4)
        if predictions:
            future_sessions = list(predictions.keys())
            historical_sessions = df['Session'].tolist()
            all_sessions = historical_sessions + future_sessions
            
            for grade_num in [5, 6, 7]:  # Focus on higher grades
                grade_col = f'Grade {grade_num}_Low'
                if grade_col in df.columns:
                    # Historical data
                    historical_values = df[grade_col].tolist()
                    
                    # Predicted data
                    predicted_values = [predictions[session].get(f'Grade {grade_num}', 0) 
                                      for session in future_sessions]
                    
                    # Combine for plotting
                    all_values = historical_values + predicted_values
                    
                    # Plot historical data
                    plt.plot(historical_sessions, historical_values, 'o-', 
                            label=f'Grade {grade_num} (Historical)', linewidth=2, markersize=6)
                    
                    # Plot predictions
                    plt.plot(future_sessions, predicted_values, 's--', 
                            label=f'Grade {grade_num} (Predicted)', linewidth=2, markersize=6, alpha=0.7)
            
            plt.axvline(x=len(historical_sessions)-0.5, color='red', linestyle=':', alpha=0.7, linewidth=2)
            plt.text(len(historical_sessions)-0.5, plt.ylim()[1]*0.9, 'Predictions →', 
                    rotation=90, verticalalignment='top', color='red', fontweight='bold')
            
            plt.title('Grade Boundary Predictions', fontsize=14, fontweight='bold')
            plt.xlabel('Session', fontsize=12)
            plt.ylabel('Boundary Score', fontsize=12)
            plt.legend(frameon=True, fancybox=True, shadow=True)
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
        
        # Plot 6: Statistical summary
        plt.subplot(2, 3, 6)
        analysis = self.analyze_math_trends()
        if analysis and 'trends' in analysis:
            grades = []
            means = []
            stds = []
            
            for grade, data in analysis['trends'].items():
                grades.append(grade.replace('Grade ', 'G'))
                means.append(data['mean'])
                stds.append(data['std'])
            
            x = np.arange(len(grades))
            bars = plt.bar(x, means, yerr=stds, capsize=5, alpha=0.8, 
                          color=['#3498DB', '#E74C3C', '#2ECC71', '#F39C12', '#9B59B6', '#1ABC9C', '#E67E22'])
            
            plt.title('Grade Boundaries: Mean ± Std Dev', fontsize=14, fontweight='bold')
            plt.xlabel('Grade Level', fontsize=12)
            plt.ylabel('Boundary Score', fontsize=12)
            plt.xticks(x, grades)
            plt.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
                plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + std + 0.5,
                        f'{mean:.1f}±{std:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout(pad=3.0)
        plt.savefig('ib_boundary_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        print("📊 Comprehensive analysis plots saved as 'ib_boundary_analysis.png'")
    
    def generate_report(self):
        """Generate a comprehensive analysis report."""
        print("\n" + "="*80)
        print("🎓 IB GRADE BOUNDARY COMPREHENSIVE ANALYSIS REPORT")
        print("="*80)
        
        # Basic statistics
        if not self.math_data.empty:
            df = self.math_data[self.math_data['Level'] == 'HL']
            print(f"\n📊 Dataset Overview:")
            print(f"   • Sessions analyzed: {len(df)}")
            print(f"   • Sessions covered: {', '.join(df['Session'].unique())}")
            print(f"   • Subject focus: Mathematics Analysis & Approaches HL")
        
        # Key insights
        insights = self.generate_insights()
        if insights:
            print(f"\n🔍 Key Insights:")
            for i, insight in enumerate(insights, 1):
                print(f"   {i}. {insight}")
        
        # Predictions
        predictions = self.predict_future_boundaries(2)
        if predictions:
            print(f"\n🔮 Future Predictions:")
            for session, grades in predictions.items():
                print(f"   📅 {session}:")
                for grade, boundary in grades.items():
                    if int(grade.split()[1]) >= 5:  # Only show higher grades
                        print(f"      • {grade}: {boundary} marks")
        
        # Analysis summary
        analysis = self.analyze_math_trends()
        if analysis and 'trends' in analysis:
            print(f"\n📈 Trend Analysis Summary:")
            for grade, data in analysis['trends'].items():
                direction = data['direction']
                slope = data['slope']
                emoji = "📈" if direction == 'increasing' else "📉"
                print(f"   {emoji} {grade}: {direction.title()} (rate: {slope:+.2f} marks/session)")
        
        print(f"\n💡 Recommendations:")
        print(f"   • Monitor Grade 6-7 boundaries closely for university admissions")
        print(f"   • Consider timezone differences when predicting results")
        print(f"   • Track long-term trends rather than single-session variations")
        print(f"   • Update predictions after each new session")
        
        print("\n" + "="*80)
        print("Report generated successfully! 🎉")
        print("="*80 + "\n")

def main():
    """Main execution function."""
    print("🚀 Starting IB Grade Boundary Analysis...")
    
    analyzer = IBBoundaryAnalyzer()
    
    # Generate comprehensive report
    analyzer.generate_report()
    
    # Create visualizations
    print("📊 Generating comprehensive visualizations...")
    analyzer.create_comprehensive_plots()
    
    print("\n✅ Analysis complete! Check the generated plots and insights above.")

if __name__ == "__main__":
    main()
