#!/usr/bin/env python3
"""
Medical Knowledge Graph Visualizer

Creates interactive network visualizations of the medical knowledge graph
stored in MongoDB, similar to Neo4j's browser interface.
"""

import os
import sys
from collections import defaultdict, Counter
from typing import Dict, List, Any, Tuple
import json

# Visualization libraries
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd

# Add src to path for imports
project_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.join(project_root, 'src'))
sys.path.insert(0, project_root)

from db.mongo_client import MongoDBClient
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MedicalGraphVisualizer:
    """Visualizes the medical knowledge graph stored in MongoDB."""
    
    def __init__(self):
        self.client = MongoDBClient()
        self.db = self.client.database
        self.kg_collection = self.db.kg
        
        # Color mapping for different entity types
        self.entity_colors = {
            'Condition': '#FF6B6B',           # Red
            'Drug_Class': '#4ECDC4',          # Teal
            'Medication': '#45B7D1',          # Blue
            'Patient_Group': '#96CEB4',       # Green
            'Age_Criteria': '#FECA57',        # Yellow
            'Ethnicity_Criteria': '#FF9FF3',  # Pink
            'Treatment_Algorithm': '#54A0FF', # Blue
            'Clinical_Decision': '#5F27CD',   # Purple
            'Monitoring': '#00D2D3',          # Cyan
            'Target': '#FF6348',              # Orange
            'Clinical_Section': '#DDA0DD',    # Plum
            'Treatment': '#98D8C8',           # Mint
            'Side_Effect': '#F7DC6F',         # Light Yellow
            'Contraindication': '#EC7063'     # Light Red
        }
        
        # Shape mapping for entity types
        self.entity_shapes = {
            'Condition': 'circle',
            'Drug_Class': 'diamond',
            'Medication': 'square',
            'Patient_Group': 'triangle-up',
            'Age_Criteria': 'star',
            'Ethnicity_Criteria': 'hexagon',
            'Treatment_Algorithm': 'pentagon',
            'Clinical_Decision': 'octagon',
            'Monitoring': 'cross',
            'Target': 'x',
            'Clinical_Section': 'circle-open',
            'Treatment': 'square-open',
            'Side_Effect': 'diamond-open',
            'Contraindication': 'triangle-down'
        }
        
    def load_graph_data(self) -> Tuple[nx.MultiDiGraph, Dict]:
        """Load graph data from MongoDB and create NetworkX graph."""
        logger.info("Loading graph data from MongoDB...")
        
        G = nx.MultiDiGraph()
        node_metadata = {}
        
        # Load all entities from kg collection
        entities = list(self.kg_collection.find())
        logger.info(f"Loaded {len(entities)} entities")
        
        for entity in entities:
            entity_id = entity['_id']
            entity_type = entity.get('type', 'Unknown')
            attributes = entity.get('attributes', {})
            
            # Add node to graph
            G.add_node(entity_id, 
                      type=entity_type,
                      **attributes)
            
            # Store metadata for visualization
            node_metadata[entity_id] = {
                'type': entity_type,
                'attributes': attributes,
                'label': entity_id.replace('_', ' ').title()
            }
            
            # Add relationships as edges
            relationships = entity.get('relationships', {})
            if relationships and 'target_ids' in relationships:
                target_ids = relationships.get('target_ids', [])
                rel_types = relationships.get('types', [])
                rel_attributes = relationships.get('attributes', [])
                
                for i, target_id in enumerate(target_ids):
                    if i < len(rel_types):
                        rel_type = rel_types[i]
                        rel_attr = rel_attributes[i] if i < len(rel_attributes) else {}
                        
                        G.add_edge(entity_id, target_id,
                                 relationship=rel_type,
                                 **rel_attr)
        
        logger.info(f"Created graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        return G, node_metadata
    
    def create_interactive_visualization(self, G: nx.MultiDiGraph, metadata: Dict, 
                                       layout='spring', title="Medical Knowledge Graph") -> go.Figure:
        """Create interactive Plotly visualization."""
        logger.info(f"Creating interactive visualization with {layout} layout...")
        
        # Calculate layout positions
        if layout == 'spring':
            pos = nx.spring_layout(G, k=1, iterations=50, seed=42)
        elif layout == 'circular':
            pos = nx.circular_layout(G)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(G)
        else:
            pos = nx.spring_layout(G, seed=42)
        
        # Prepare node traces
        node_traces = []
        for entity_type in self.entity_colors.keys():
            nodes_of_type = [node for node, attr in G.nodes(data=True) 
                           if attr.get('type') == entity_type]
            
            if not nodes_of_type:
                continue
                
            node_x = [pos[node][0] for node in nodes_of_type if node in pos]
            node_y = [pos[node][1] for node in nodes_of_type if node in pos]
            
            # Create hover text with entity details
            hover_text = []
            for node in nodes_of_type:
                if node in pos:
                    attrs = metadata.get(node, {}).get('attributes', {})
                    hover_info = f"<b>{node}</b><br>"
                    hover_info += f"Type: {entity_type}<br>"
                    
                    # Add key attributes to hover
                    for key, value in attrs.items():
                        if key in ['name', 'source_section', 'clinical_context']:
                            if isinstance(value, list) and value:
                                hover_info += f"{key.title()}: {value[0]}<br>"
                            elif not isinstance(value, list):
                                hover_info += f"{key.title()}: {str(value)[:50]}<br>"
                    
                    hover_text.append(hover_info)
            
            # Create node trace
            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                marker=dict(
                    size=20,
                    color=self.entity_colors.get(entity_type, '#BDC3C7'),
                    line=dict(width=2, color='white'),
                    opacity=0.8
                ),
                text=[metadata.get(node, {}).get('label', node)[:15] for node in nodes_of_type if node in pos],
                textposition="middle center",
                textfont=dict(size=10, color="white"),
                hovertemplate='%{hovertext}<extra></extra>',
                hovertext=hover_text,
                name=entity_type,
                showlegend=True
            )
            node_traces.append(node_trace)
        
        # Prepare edge traces
        edge_traces = []
        edge_colors = {
            'TREATS': '#2ECC71',
            'CONTAINS': '#95A5A6', 
            'MONITORS': '#3498DB',
            'FIRST_LINE_FOR': '#E74C3C',
            'IF_NOT_TOLERATED': '#F39C12',
            'CONDITIONAL_ON': '#9B59B6',
            'ALTERNATIVE_TO': '#1ABC9C'
        }
        
        for edge in G.edges(data=True):
            source, target, data = edge
            if source in pos and target in pos:
                rel_type = data.get('relationship', 'UNKNOWN')
                
                edge_trace = go.Scatter(
                    x=[pos[source][0], pos[target][0], None],
                    y=[pos[source][1], pos[target][1], None],
                    mode='lines',
                    line=dict(
                        width=2,
                        color=edge_colors.get(rel_type, '#BDC3C7')
                    ),
                    hovertemplate=f'<b>{rel_type}</b><br>{source} → {target}<extra></extra>',
                    name=rel_type,
                    showlegend=False,
                    opacity=0.6
                )
                edge_traces.append(edge_trace)
        
        # Create figure
        fig = go.Figure(data=edge_traces + node_traces)
        
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                font=dict(size=20)
            ),
            showlegend=True,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            annotations=[ dict(
                text="Hover over nodes for details. Legend shows entity types.",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002,
                xanchor='left', yanchor='bottom',
                font=dict(color="#888", size=12)
            )],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white',
            height=800
        )
        
        return fig
    
    def create_network_statistics_dashboard(self, G: nx.MultiDiGraph) -> go.Figure:
        """Create a dashboard with network statistics."""
        logger.info("Creating network statistics dashboard...")
        
        # Calculate statistics
        node_types = Counter([data.get('type', 'Unknown') for _, data in G.nodes(data=True)])
        edge_types = Counter([data.get('relationship', 'UNKNOWN') for _, _, data in G.edges(data=True)])
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Node Types Distribution', 'Relationship Types', 
                          'Node Degree Distribution', 'Network Metrics'),
            specs=[[{'type': 'pie'}, {'type': 'bar'}],
                   [{'type': 'histogram'}, {'type': 'table'}]]
        )
        
        # Node types pie chart
        fig.add_trace(
            go.Pie(labels=list(node_types.keys()), 
                  values=list(node_types.values()),
                  marker_colors=[self.entity_colors.get(t, '#BDC3C7') for t in node_types.keys()]),
            row=1, col=1
        )
        
        # Relationship types bar chart
        fig.add_trace(
            go.Bar(x=list(edge_types.keys()), 
                  y=list(edge_types.values()),
                  marker_color='lightblue'),
            row=1, col=2
        )
        
        # Node degree distribution
        degrees = [G.degree(n) for n in G.nodes()]
        fig.add_trace(
            go.Histogram(x=degrees, nbinsx=20, marker_color='green'),
            row=2, col=1
        )
        
        # Network metrics table
        metrics = [
            ['Total Nodes', G.number_of_nodes()],
            ['Total Edges', G.number_of_edges()],
            ['Average Degree', f"{sum(degrees)/len(degrees):.2f}" if degrees else "0"],
            ['Density', f"{nx.density(G):.4f}"],
            ['Connected Components', nx.number_weakly_connected_components(G)]
        ]
        
        fig.add_trace(
            go.Table(
                header=dict(values=['Metric', 'Value']),
                cells=dict(values=list(zip(*metrics)))
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title_text="Medical Knowledge Graph Analytics",
            height=800,
            showlegend=False
        )
        
        return fig
    
    def save_interactive_html(self, fig: go.Figure, filename: str = "medical_knowledge_graph.html"):
        """Save interactive visualization as HTML file."""
        filepath = os.path.join(os.path.dirname(__file__), '..', filename)
        fig.write_html(filepath)
        logger.info(f"Interactive visualization saved to: {filepath}")
        return filepath
    
    def export_to_gephi_format(self, G: nx.MultiDiGraph, filename: str = "medical_graph.gexf"):
        """Export graph in Gephi format for advanced visualization."""
        filepath = os.path.join(os.path.dirname(__file__), '..', filename)
        nx.write_gexf(G, filepath)
        logger.info(f"Graph exported to Gephi format: {filepath}")
        return filepath
    
    def create_subgraph_visualization(self, focus_entity: str, max_depth: int = 2) -> go.Figure:
        """Create visualization focused on a specific entity and its neighborhood."""
        logger.info(f"Creating subgraph visualization for: {focus_entity}")
        
        G, metadata = self.load_graph_data()
        
        if focus_entity not in G:
            logger.error(f"Entity '{focus_entity}' not found in graph")
            return None
        
        # Create subgraph around focus entity
        subgraph_nodes = set([focus_entity])
        current_nodes = set([focus_entity])
        
        for depth in range(max_depth):
            next_nodes = set()
            for node in current_nodes:
                # Add neighbors (both incoming and outgoing)
                next_nodes.update(G.successors(node))
                next_nodes.update(G.predecessors(node))
            
            subgraph_nodes.update(next_nodes)
            current_nodes = next_nodes
            
            if not current_nodes:
                break
        
        subgraph = G.subgraph(subgraph_nodes).copy()
        
        return self.create_interactive_visualization(
            subgraph, metadata, 
            title=f"Subgraph: {focus_entity} (depth {max_depth})"
        )

def main():
    """Main function to create and display visualizations."""
    try:
        visualizer = MedicalGraphVisualizer()
        
        # Load graph data
        G, metadata = visualizer.load_graph_data()
        
        if G.number_of_nodes() == 0:
            logger.warning("No graph data found. Please run graph building scripts first.")
            return
        
        # Create main visualization
        print("🎨 Creating interactive network visualization...")
        main_fig = visualizer.create_interactive_visualization(G, metadata)
        main_html = visualizer.save_interactive_html(main_fig, "medical_knowledge_graph.html")
        
        # Create statistics dashboard
        print("📊 Creating network statistics dashboard...")
        stats_fig = visualizer.create_network_statistics_dashboard(G)
        stats_html = visualizer.save_interactive_html(stats_fig, "graph_analytics_dashboard.html")
        
        # Export to Gephi format
        print("💾 Exporting to Gephi format...")
        gephi_file = visualizer.export_to_gephi_format(G)
        
        # Create focused subgraph (example)
        if 'hypertension' in G:
            print("🔍 Creating focused subgraph for 'hypertension'...")
            hypertension_fig = visualizer.create_subgraph_visualization('hypertension', max_depth=2)
            if hypertension_fig:
                visualizer.save_interactive_html(hypertension_fig, "hypertension_subgraph.html")
        
        print("\n" + "="*60)
        print("✅ VISUALIZATION COMPLETE!")
        print("="*60)
        print(f"📊 Main Graph: file://{os.path.abspath(main_html)}")
        print(f"📈 Analytics: file://{os.path.abspath(stats_html)}")
        print(f"💾 Gephi Export: {os.path.abspath(gephi_file)}")
        print("\nOpen the HTML files in your browser for interactive exploration!")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Visualization failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()