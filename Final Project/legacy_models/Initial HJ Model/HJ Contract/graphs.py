import plotly.graph_objects as go


def plot_results(test_dates, y_test, predictions, capital_performance_over_time_2, r2_values):
    fig = go.Figure()

    # Add actual spread values (left y-axis)
    fig.add_trace(go.Scatter(
        x=test_dates,
        y=y_test,
        mode='lines+markers',
        name='Actual Spread',
        line=dict(color='blue'),
        marker=dict(size=8),
        yaxis="y1"
    ))

    # Add predicted spread values (left y-axis)
    fig.add_trace(go.Scatter(
        x=test_dates,
        y=predictions,
        mode='lines+markers',
        name='Predicted Spread',
        line=dict(color='red', dash='dash'),
        marker=dict(size=8),
        yaxis="y1"
    ))

    # Add Portfolio performance over time (right y-axis)
    fig.add_trace(go.Scatter(
        x=test_dates[1:],
        y=capital_performance_over_time_2[1:],
        mode='lines+markers',
        name='Portfolio Performance (%)',
        line=dict(color='orange'),
        marker=dict(size=8),
        yaxis="y2"
    ))

    # Add R² values over time (third y-axis)
    fig.add_trace(go.Scatter(
        x=test_dates[1:],
        y=r2_values,
        mode='lines',
        name='R² Over Time',
        line=dict(color='purple', dash='dot'),
        opacity=0.6,
        yaxis="y3"
    ))

    # Update layout to include triple y-axes
    fig.update_layout(
        title="Actual vs Predicted Spread, Portfolio Performances, and R² Over Time",
        xaxis_title="Date",
        yaxis=dict(title="Spread ($)", side="left", showgrid=False),
        yaxis2=dict(title="Portfolio Performance (%)",
                    overlaying="y", side="right", showgrid=False),
        yaxis3=dict(
            title="R² Value",
            overlaying="y",
            side="right",
            position=0.95,
            tickmode="auto",
            tickformat=".2f",
            showgrid=False,
            tickfont=dict(color="purple"),
            titlefont=dict(color="purple")
        ),
        legend_title="Legend"
    )

    # Show plot
    fig.show()
