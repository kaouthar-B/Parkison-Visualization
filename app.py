from dash import dcc, html
import dash
import pandas as pd
import plotly.express as px
from dash.dependencies import Input, Output
import ast
import plotly.graph_objects as go
import numpy as np


# Charger les deux datasets
df_parkinsons = pd.read_csv('data/parkinsons_data.csv')
df_neural = pd.read_csv("data/neural_activity_with_target.csv")

# Prétraitement de `parkinsons_data.csv`
df_parkinsons['AgeGroup'] = pd.cut(
    df_parkinsons['Age'], bins=[50, 60, 70, 80, 90], labels=['50-60', '60-70', '70-80', '80-90']
)
df_parkinsons['AgeGroup'] = df_parkinsons['AgeGroup'].cat.add_categories('Unknown')
df_parkinsons.fillna({'AgeGroup': 'Unknown'}, inplace=True)

# Prétraitement de `neural_activity_with_target.csv`
for col in ['Raw_Signal', 'Magnitude', 'Phase']:
    df_neural[col] = df_neural[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
# Exemple de color_map pour Sunburst
color_map = {'Diagnosis1': 'blue', 'Diagnosis2': 'green', 'Diagnosis3': 'red'}

# Initialiser l'application Dash
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "Dashboard Intégré Parkinson"

# Layout principal
app.layout = html.Div([
    html.H1("Dashboard Parkinson", style={'textAlign': 'center'}),
    dcc.Tabs([
        # Visualisation des données `parkinsons_data.csv`
        dcc.Tab(label="Données Parkinson", children=[
            dcc.Tabs([
                # Onglet 1 : Répartition Âge et Genre
                dcc.Tab(label="Répartition Âge et Genre", children=[
                    dcc.Graph(
                        id='age-gender',
                        figure=px.histogram(df_parkinsons, x="Age", color="Gender", title="Répartition Âge et Genre",
                                            color_discrete_sequence=px.colors.sequential.Viridis)
                    )
                ]),
                # Onglet 2 : Capacités de Mouvement
                dcc.Tab(label="Capacités de Mouvement", children=[
                    dcc.Graph(
                        id='movement-capacity',
                        figure=px.box(df_parkinsons, x='Diagnosis', y='FunctionalAssessment', color='Diagnosis',
                                      title="Functional Assessment par Diagnostic")
                    ),
                    dcc.Graph(
                        id='age-diagnosis',
                        figure=px.box(df_parkinsons, x='Diagnosis', y='Age', color='Diagnosis',
                                      title="Âge par Diagnostic de Parkinson")
                    )
                ]),
                # Onglet 3 : Symptômes Fréquents
                dcc.Tab(label="Symptômes Fréquents", children=[
                    dcc.Graph(
                        id='symptom-frequencies',
                        figure=px.histogram(df_parkinsons.melt(id_vars=['Diagnosis'], value_vars=[
                            'Tremor', 'Rigidity', 'Bradykinesia', 'PosturalInstability',
                            'SpeechProblems', 'SleepDisorders', 'Constipation'
                        ], var_name='Symptom', value_name='Presence'),
                        x='Symptom', color='Presence', barmode='group', facet_col='Diagnosis',
                        title="Fréquence des Symptômes par Diagnostic")
                    )
                ]),
                # Onglet 4 : Corrélations
                dcc.Tab(label="Corrélations", children=[
                    dcc.Graph(
                        id='correlation-matrix',
                        figure=px.imshow(df_parkinsons[['Tremor', 'Rigidity', 'Bradykinesia', 'PosturalInstability',
                                                        'SpeechProblems', 'SleepDisorders', 'Constipation', 'Diagnosis']].corr(),
                                         title="Matrice de Corrélation")
                    )
                ]),
                # Onglet 5 : Nuage de Points 3D
                dcc.Tab(label="Nuage de Points 3D", children=[
                    dcc.Graph(
                        id='3d-scatter',
                        figure=px.scatter_3d(df_parkinsons, x='SystolicBP', y='PhysicalActivity', z='FunctionalAssessment',
                                             color='Diagnosis', title="Nuage de Points 3D")
                    )
                ]),
                # Onglet 6 : Analyses
                dcc.Tab(label="Analyses", children=[
                    dcc.Graph(
                        id='smoking-alcohol',
                        figure=px.bar(df_parkinsons, x='Smoking', y='AlcoholConsumption', color='Diagnosis',
                                      title="Smoking vs Alcohol Consumption par Diagnostic",
                                      barmode='group')
                    ),
                    dcc.Graph(
                        id='symptom-stack',
                        figure=px.bar(df_parkinsons.groupby('Diagnosis')[['Tremor', 'Rigidity', 'Bradykinesia',
                                                                          'PosturalInstability', 'SpeechProblems']].sum().reset_index(),
                                      x='Diagnosis', y=['Tremor', 'Rigidity', 'Bradykinesia', 'PosturalInstability', 'SpeechProblems'],
                                      title="Fréquence des Symptômes par Diagnostic", barmode='stack')
                    )
                ]),
                # Onglet 7 : Âge par Diagnostic
                dcc.Tab(label="Âge par Diagnostic", children=[
                    dcc.Graph(
                        id='age-diagnosis-distribution',
                        figure=px.histogram(df_parkinsons, x="Age", color="Diagnosis",
                                            title="Distribution de l'Âge par Diagnostic",
                                            color_discrete_sequence=px.colors.qualitative.Set2)
                    )
                ]),
                # Onglet 8 : Tremblement vs Diagnostic
                dcc.Tab(label="Tremblement vs Diagnostic", children=[
                    dcc.Graph(
                        id='tremor-diagnosis',
                        figure=px.histogram(df_parkinsons, x='Tremor', color='Diagnosis', barmode='stack',
                                            title="Tremblement vs Diagnostic")
                    )
                ]),
                # Onglet 9 : Répartition Démographique
                dcc.Tab(label="Répartition Démographique", children=[
                    dcc.Graph(
                        id='demographic-breakdown',
                        figure=px.sunburst(df_parkinsons, path=['Gender', 'AgeGroup', 'Diagnosis'], color='Diagnosis',
                                           title="Répartition Démographique par Diagnostic",
                                           color_discrete_map=color_map)
                    )
                ])
            ])
        ]),


  # Visualisation des données `neural_activity_with_target.csv`
        dcc.Tab(label="Données Neural Activity", children=[
            html.Div([
                html.Button("Signaux des Patients", id='signals-btn', n_clicks=0),
                html.Button("Variables de Base", id='base-vars-btn', n_clicks=0),
                html.Button("Cible (Target)", id='target-btn', n_clicks=0),
            ], style={'display': 'flex', 'gap': '10px', 'margin': '20px'}),
            html.Div(id='dynamic-content'),
        ]),
    ])
])

# Callbacks pour la partie `neural_activity_with_target.csv`
@app.callback(
    Output('dynamic-content', 'children'),
    [Input('signals-btn', 'n_clicks'),
     Input('base-vars-btn', 'n_clicks'),
     Input('target-btn', 'n_clicks')]
)
def update_content(signals_clicks, base_vars_clicks, target_clicks):
    ctx = dash.callback_context
    if not ctx.triggered:
        button_id = 'signals-btn'
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'signals-btn':
        return html.Div([
            html.H3("Signal Brut - Sélectionner un patient"),
            dcc.Dropdown(
                id='target-filter-dropdown',
                options=[{'label': str(target), 'value': target} for target in df_neural['Target'].unique()],
                value=df_neural['Target'].unique().tolist(),
                multi=True,
                style={'width': '50%'}
            ),
            dcc.Dropdown(
                id='patient-dropdown',
                options=[],
                value=[],
                multi=True,
                style={'width': '50%'}
            ),
            dcc.Graph(id='signal-graph'),
            html.Div(id='patient-info', style={'marginTop': '20px', 'fontSize': '16px'})
        ])
    elif button_id == 'base-vars-btn':
        return html.Div([
            html.H3("Variables de Base"),
            dcc.Graph(id='age-distribution'),
            dcc.Graph(id='gender-distribution'),
            dcc.Graph(id='clinical-stage-distribution'),
            dcc.Graph(id='condition-label-distribution'),
        ])
    elif button_id == 'target-btn':
        return html.Div([
            html.H3("Cible de Traitement"),
            dcc.Graph(id='target-distribution'),
        ])

@app.callback(
    Output('patient-dropdown', 'options'),
    Input('target-filter-dropdown', 'value')
)
def update_patient_dropdown(selected_targets):
    filtered_df = df_neural[df_neural['Target'].isin(selected_targets)]
    return [{'label': f"Patient {i}", 'value': i} for i in filtered_df.index]

@app.callback(
    [Output('signal-graph', 'figure'),
     Output('patient-info', 'children')],
    Input('patient-dropdown', 'value')
)
def update_signal_and_info(patient_ids):
    if not patient_ids:
        return {}, ""

    data = []
    patient_info = []
    for patient_id in patient_ids:
        signal = df_neural.loc[patient_id, 'Raw_Signal'][:2000]
        data.append(go.Scatter(x=np.arange(len(signal)), y=signal, mode='lines', name=f'Signal Patient {patient_id}'))
        
        clinical_stage = df_neural.loc[patient_id, 'Clinical_Stage']
        condition_label = df_neural.loc[patient_id, 'Condition_Label']
        patient_info.append(f"Patient {patient_id} - Stage Clinique: {clinical_stage}, Label de Condition: {condition_label}")
    
    figure = {
        'data': data,
        'layout': go.Layout(
            title='Signaux Bruts',
            xaxis={'title': 'Temps'},
            yaxis={'title': 'Amplitude'}
        )
    }
    return figure, "<br>".join(patient_info)

@app.callback(
    Output('age-distribution', 'figure'),
    Input('base-vars-btn', 'n_clicks')
)
def update_age_distribution(_):
    return {
        'data': [go.Histogram(x=df_neural['Age'], nbinsx=10, marker=dict(color='blue'))],
        'layout': go.Layout(
            title='Répartition de l\'Âge',
            xaxis={'title': 'Âge'},
            yaxis={'title': 'Nombre de Patients'}
        )
    }

@app.callback(
    Output('gender-distribution', 'figure'),
    Input('base-vars-btn', 'n_clicks')
)
def update_gender_distribution(_):
    gender_counts = df_neural['Gender'].value_counts()
    return {
        'data': [go.Pie(labels=gender_counts.index, values=gender_counts.values, hole=0.3)],
        'layout': go.Layout(title='Répartition des Genres')
    }

@app.callback(
    Output('clinical-stage-distribution', 'figure'),
    Input('base-vars-btn', 'n_clicks')
)
def update_clinical_stage_distribution(_):
    stage_counts = df_neural['Clinical_Stage'].value_counts()
    return {
        'data': [go.Pie(labels=stage_counts.index, values=stage_counts.values, hole=0.3)],
        'layout': go.Layout(title='Répartition des Stades Cliniques')
    }

@app.callback(
    Output('condition-label-distribution', 'figure'),
    Input('base-vars-btn', 'n_clicks')
)
def update_condition_label_distribution(_):
    condition_counts = df_neural['Condition_Label'].value_counts()
    return {
        'data': [go.Bar(x=condition_counts.index, y=condition_counts.values)],
        'layout': go.Layout(
            title='Répartition des Labels de Condition',
            xaxis={'title': 'Condition'},
            yaxis={'title': 'Nombre de Patients'}
        )
    }

@app.callback(
    Output('target-distribution', 'figure'),
    Input('target-btn', 'n_clicks')
)
def update_target_distribution(_):
    target_counts = df_neural['Target'].value_counts()
    return {
        'data': [go.Pie(labels=target_counts.index, values=target_counts.values, hole=0.3)],
        'layout': go.Layout(title='Répartition des Cibles de Traitement')
    }

# Lancer l'application
if __name__ == '__main__':
    app.run_server(debug=True)