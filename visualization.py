import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

def plot_results(results, use_plotly=True, show_terminal=False, title="Training Results", smoothing=0, evaluation_results=None):
    """
    Visualize the resource allocation results using Plotly.

    Args:
        results: List of results from resource allocation simulation
        use_plotly: Whether to generate a Plotly visualization
        show_terminal: Whether to show terminal visualization
        title: Title for the visualization
        smoothing: Window size for smoothing the results (0 = no smoothing)
        evaluation_results: Optional evaluation results to overlay
    
    Returns:
        fig: Plotly figure object if use_plotly=True, otherwise None
    """
    # Apply smoothing if requested
    if smoothing > 0 and len(results) > smoothing:
        smoothed_results = np.convolve(results, np.ones(smoothing)/smoothing, mode='valid')
        plot_data = smoothed_results.tolist()
    else:
        plot_data = results
    
    # Show terminal visualization if requested
    if show_terminal:
        plot_results_terminal(results, title=title)
    
    # Generate Plotly visualization
    if use_plotly:
        try:
            # Create a single plot with improved styling
            fig = go.Figure()
            
            # Add line trace for rewards
            fig.add_trace(go.Scatter(
                x=list(range(1, len(results) + 1)),
                y=results,
                mode='lines',
                name='Training Reward',
                line=dict(color='rgba(65, 105, 225, 0.8)', width=2)
            ))
            
            # Add running average if we have enough data
            if len(results) >= 5:
                window = min(5, len(results) // 5)
                running_avg = np.convolve(results, np.ones(window)/window, mode='valid')
                x_running = list(range(window, len(results) + 1))
                
                fig.add_trace(go.Scatter(
                    x=x_running,
                    y=running_avg,
                    mode='lines',
                    name='Trend',
                    line=dict(color='rgba(255, 87, 51, 0.9)', width=3)
                ))
            
            # Add evaluation results if provided
            if evaluation_results:
                # Calculate positions for evaluation results (at the end)
                eval_x = list(range(len(results) - len(evaluation_results) + 1, len(results) + 1))
                
                # Add evaluation points
                fig.add_trace(go.Scatter(
                    x=eval_x,
                    y=evaluation_results,
                    mode='markers',
                    name='Evaluation',
                    marker=dict(
                        color='rgba(0, 200, 81, 1.0)',
                        size=10,
                        symbol='diamond',
                        line=dict(color='rgba(0, 0, 0, 0.5)', width=1)
                    )
                ))
                
                # Add average evaluation line
                avg_eval = sum(evaluation_results) / len(evaluation_results)
                fig.add_trace(go.Scatter(
                    x=[eval_x[0], eval_x[-1]],
                    y=[avg_eval, avg_eval],
                    mode='lines',
                    name=f'Avg Eval: {avg_eval:.2f}',
                    line=dict(color='rgba(0, 176, 80, 0.8)', width=2, dash='dash')
                ))
            
            # Update layout with improved styling
            fig.update_layout(
                title={
                    'text': title,
                    'font': {'size': 24, 'family': 'Arial, sans-serif'}
                },
                xaxis_title={
                    'text': 'Episode',
                    'font': {'size': 16, 'family': 'Arial, sans-serif'}
                },
                yaxis_title={
                    'text': 'Reward',
                    'font': {'size': 16, 'family': 'Arial, sans-serif'}
                },
                template='plotly_white',
                legend={
                    'x': 0.02, 
                    'y': 0.98, 
                    'bgcolor': 'rgba(255, 255, 255, 0.8)',
                    'bordercolor': 'rgba(0, 0, 0, 0.1)',
                    'borderwidth': 1
                },
                margin=dict(l=40, r=40, t=80, b=40),
                plot_bgcolor='rgba(240, 240, 240, 0.5)',
                width=900,
                height=500,
                # Add a border around the plot
                shapes=[
                    dict(
                        type='rect',
                        xref='paper',
                        yref='paper',
                        x0=0,
                        y0=0,
                        x1=1,
                        y1=1,
                        line=dict(color='rgba(0, 0, 0, 0.2)', width=1)
                    )
                ]
            )
            
            # Show the plot
            fig.show()
            return fig
        except Exception as e:
            print(f"Failed to plot results with Plotly: {e}")
            return None
    
    return None

def plot_results_terminal(results, title="Training Results"):
    """
    Visualize the resource allocation results in the terminal using ASCII art.

    Args:
        results: List of results from resource allocation simulation
        title: Title for the visualization
    """
    if not results:
        print("No results to display.")
        return
    
    print(f"\n=== {title} ===")
    print(f"Episodes: {len(results)}")
    print(f"Final: {results[-1]:.2f}")
    print(f"Max: {max(results):.2f}")
    print(f"Min: {min(results):.2f}")
    print(f"Average: {sum(results)/len(results):.2f}")
    
    # Calculate trend
    if len(results) > 1:
        x = np.array(range(len(results)))
        slope, _ = np.polyfit(x, results, 1)
        print(f"Trend: {'Improving' if slope > 0 else 'Declining'}")

def plot_comparative_results(result_sets, labels, use_plotly=True, show_terminal=False, title="Comparison"):
    """
    Visualize multiple result sets for comparison.
    
    Args:
        result_sets: List of result lists from different runs
        labels: List of labels for each result set
        use_plotly: Whether to generate a Plotly visualization
        show_terminal: Whether to show terminal visualization
        title: Title for the visualization
        
    Returns:
        fig: Plotly figure object if use_plotly=True, otherwise None
    """
    if not result_sets or len(result_sets) != len(labels):
        print("Invalid result sets or labels provided.")
        return None
    
    # Terminal visualization
    if show_terminal:
        print(f"\n=== {title} ===")
        
        # Print summary statistics for each result set
        for i, (results, label) in enumerate(zip(result_sets, labels)):
            if not results:
                continue
                
            print(f"\n--- {label} ---")
            print(f"Episodes: {len(results)}")
            print(f"Final: {results[-1]:.2f}")
            print(f"Max: {max(results):.2f}")
            print(f"Min: {min(results):.2f}")
            print(f"Average: {sum(results)/len(results):.2f}")
    
    # Plotly visualization
    if use_plotly:
        try:
            fig = go.Figure()
            
            # Add a trace for each result set
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
            
            for i, (results, label) in enumerate(zip(result_sets, labels)):
                if not results:
                    continue
                    
                color = colors[i % len(colors)]
                
                # Add line trace
                fig.add_trace(go.Scatter(
                    x=list(range(1, len(results) + 1)),
                    y=results,
                    mode='lines',
                    name=label,
                    line=dict(color=color, width=2)
                ))
            
            # Update layout
            fig.update_layout(
                title=title,
                xaxis_title='Episode',
                yaxis_title='Reward',
                template='plotly_white',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=40, r=40, t=50, b=40)
            )
            
            # Show the plot
            fig.show()
            return fig
        except Exception as e:
            print(f"Failed to plot comparative results with Plotly: {e}")
            return None
    
    return None

def main():
    """Test visualization functions with sample data"""
    import random
    
    print("Testing improved visualization module...")
    
    # Generate some sample data
    # Trending upward
    up_trend = [20 + i * 0.5 + random.uniform(-3, 3) for i in range(30)]
    
    # Test single result visualization
    plot_results(up_trend, title="Training Progress")
    
    print("Visualization test completed.")

if __name__ == "__main__":
    main()