import base64
from io import BytesIO

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import dash_html_components as html
import multidict
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud

from plot.tag import ENTITY_TAGS


def find_nearby_posts(previous_post_ids: list, list_posts: list, same_category: bool=False, radius: int=2, topk: int=5):
    """

    :param previous_post_ids:
    :param list_posts:
    :param same_category:
    :param radius:
    :param topk:
    :return:
    """
    pass


def entity_span(entity, type_, color):
    meta = ENTITY_TAGS.get(type_, {})

    return html.Mark(children=[
        entity,
        html.Span(
            f"{meta.get('alias', 'OTH')}",
            className="annotation",
            style={
                "display": "inline-block",
                "boxSizing": "border-box",
                "fontSize": "0.55em",
                "fontWeight": "bold",
                "background": "#fff",
                "padding": "0.35em",
                "verticalAlign": "middle",
                "borderRadius": "0.35em",
                "marginBottom": "0px",
                "marginRight": "0.15em",
                "marginLeft": "0.5em"
            }
        )
    ], style={"backgroundColor": meta.get("color", "grey"), "color": "black"})


def highlighted_markup(text, tags):
    children = []
    start_index = 0
    for tag in tags:
        next_offset = tag['offset']
        children.append(text[start_index:next_offset])
        children.append(entity_span(tag['text'], tag['type_'], "red"))
        start_index = next_offset + tag['len']
    return html.Div(
        children=children,
        style={
            "width": "100%",
            # "height": "470px",
            "overflowY": "auto",
            "overflowWrap": "break-word",
            "marginBottom": "10px"
        }
    )


def create_pie(names, values):
    colors = {
        'Negative': '#E33a34',
        'Positive': '#347aeb',
        'Neutral': '#439B4E'
    }

    df = pd.DataFrame([names, values]).T
    df.columns = ['name', 'value']
    df['color'] = df['name'].map(colors)
    df['explode'] = [0.05, 0.0005, 0.0005]

    fig = Figure(figsize=(16, 9))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('equal')

    _, texts, _ = ax.pie(
        df['value'],
        colors=df['color'],
        autopct='%1.f%%',
        wedgeprops=dict(width=.7),
        labels=df['name'],
        textprops={'color': "w", "fontsize": 25},
        normalize=True,
        startangle=90,
        pctdistance=0.7,
        explode=df['explode'],
        # shadow=True
    )
    [text.set_color('black') for text in texts]

    buf = BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    return 'data:image/png;base64,{}'.format(base64.b64encode(buf.getvalue()).decode())


def create_funnel_area(score, values, names):
    colors = {
        'Negative': '#E33a34',
        'Positive': '#439B4E',
        'Neutral': 'grey'
    }

    df = pd.DataFrame([names, values]).T
    df.columns = ['name', 'value']
    df['color'] = df['name'].map(colors)

    #flag = 'neutral'
    #if score > 0:
    #    flag = 'positive'
    #elif score < 0:
    #    flag = 'negative'

    fig = go.Figure(go.Funnelarea(
        values=df['value'],
        text=df['name'],
        labels=df['name'],
        marker={'colors': df['color']},
        textfont={"size": 16, "color": "white"},
        hovertemplate=f'Value:' + '%{value}<extra></extra>'
    ))

    fig.update_layout(
        # title=f"Text has {flag} sentiment with overall score: <b>{round(score * 100, 3)}%</b>",
        margin=dict(t=1),
        # showlegend=False,
        legend=dict(orientation="h", yanchor="top",
                    y=1.1,
                    xanchor="center",
                    x=0.5),
    )
    return fig


def create_3d_graph(circles, xylim=0.4, gridN=200, spreadN=1.66):
    annotations = []
    grid = np.zeros((gridN, gridN))

    for i in range(circles.shape[0]):
        if not circles['delete'][i]:
            grid_circle_x = int((circles['x'][i] + xylim) / (2 * xylim) * gridN)
            grid_circle_y = int((circles['y'][i] + xylim) / (2 * xylim) * gridN)
            grid_radius = int(circles['size'][i] / (2 * xylim) * gridN * spreadN)
            annotations.append(dict(
                showarrow=False,
                x=grid_circle_y,
                y=grid_circle_x,
                z=circles['height'][i],
                text=circles['entity'][i],
                xanchor="auto",
                xshift=0,
                yanchor="auto",
                yshift=0
            ))
            for i1 in range(max(0, grid_circle_x - grid_radius), min(gridN, grid_circle_x + grid_radius)):
                for j1 in range(max(0, grid_circle_y - grid_radius), min(gridN, grid_circle_y + grid_radius)):
                    dist = np.sqrt(np.power(i1 - grid_circle_x, 2) + np.power(j1 - grid_circle_y, 2))
                    if dist < grid_radius:
                        val = (1 - dist * dist / (grid_radius * grid_radius)) * circles['height'][i]
                        if val > grid[i1, j1]:
                            grid[i1, j1] = val

    # sh_0, sh_1 = grid.shape
    # x, y = np.linspace(0, 1, sh_0), np.linspace(0, 1, sh_1)

    fig = go.Figure(
        data=[go.Surface(
            z=grid,
            # colorbar=dict(
            #     title="Colorbar",
            #     # thickness=10,
            #     lenmode="pixels",
            #     # len=50,
            #     yanchor="top",
            #     y=1,
            #     dtick=3
            # ),
            colorscale="ylgnbu",
            showscale=False,

        )],
        layout=go.Layout(
            # autosize=False,
            scene=dict(
                xaxis=dict(
                    backgroundcolor="rgb(200, 200, 230)",
                    gridcolor="white",
                    showbackground=True,
                    zerolinecolor="white", ),
                yaxis=dict(
                    backgroundcolor="rgb(230, 200,230)",
                    gridcolor="white",
                    showbackground=True,
                    zerolinecolor="white"),
                zaxis=dict(
                    backgroundcolor="rgb(230, 230,200)",
                    gridcolor="white",
                    showbackground=True,
                    zerolinecolor="white", ),
                annotations=annotations
            ),
            # width=1000,
            height=530,
            showlegend=False,

            margin=dict(
                l=50,
                r=50,
                b=50,
                t=1
            )
        )
    )

    camera = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
        # eye=dict(x=2, y=2, z=0.1)
    )

    fig.update_layout(
        scene_camera=camera,
        # title='3D graph',
        # scene_camera_eye=dict(x=1.87, y=0.88, z=-0.64),
        # font_family="Courier New",
        # font_color="red",
        # title_font_family="Times New Roman",
        # title_font_color="black",
        # coloraxis_colorbar=dict(
        #     # title="Number of Bills per Cell",
        #     thicknessmode="pixels", thickness=10,
        #     lenmode="pixels", len=10,
        #     yanchor="top", y=1,
        #     dtick=5
        # )
        # legend_title_font_color="green",
        # autosize=False,
        # width=1000,
        # height=600,
        margin=dict(l=10, r=10, t=30, b=50)
    )

    # fig.update_traces(contours_z=dict(show=True, usecolormap=True,
    #                                   highlightcolor="limegreen", project_z=True))

    return fig.to_dict()


def create_bar_graph(df, top=10):
    fig = px.bar(
        df.sort_values('salience', ascending=False).head(top),
        x='entity',
        y='salience',
        labels={
            'entity': 'Entity',
            'salience': 'Salience'
        }
    )
    return fig.to_dict()


def create_2d_graph(df, text_col='entity', overlap=None, text_pos='top center', width=None, height=None, mobile=False, **kwargs):
    tmp = df.copy()
    if not tmp.empty and not overlap:
        tmp = tmp[~tmp.delete]

    if mobile:
        scale = width / 1000
    else:
        scale = 1

    tmp['font_size'] = np.ceil(scale * 13).astype(np.int8)
    if mobile and width < height:
        a, b = tmp['x'].copy(True), tmp['y'].copy(True)
        tmp['x'], tmp['y'] = b, a
        tmp['size'] = tmp['size'] * scale
        tmp['font_size'] = np.ceil(scale * 24).astype(np.int8)

    max_size = tmp['size'].max()

    sizeref = 2. * max_size / ((100 * scale) ** 2)
    fig = go.Figure(data=[go.Scatter(
        x=tmp['x'],
        y=tmp['y'],
        mode='markers+text',

        text=tmp[text_col],
        textposition=text_pos,
        marker_size=tmp['size'],
        marker=dict(
            sizemode='area',
            color=tmp['size'],
            sizeref=sizeref,
            line_width=2,
            showscale=False,
            # colorscale="peach"
        ),
        opacity=0.95,
        hovertemplate="%{text}<extra></extra>"
    )])

    fig.update_layout(
        # title=title,
        # autosize=False,
        # width=700,
        height=height,
        margin=dict(l=3, r=3, t=3, b=3),
        hoverlabel_align='right'
    )

    if mobile:
        fig.update_layout(dragmode=False)
        fig.update_yaxes(showticklabels=False, automargin=True)
        fig.update_xaxes(showticklabels=True, automargin=True)

    return fig.to_dict()


def create_2d_graph2(df: pd.DataFrame, text_col='entity_s', overlap=None, text_pos='top center', width=None,
                     height=None, mobile=None, **kwargs):
    tmp = df.copy()

    if not tmp.empty and not overlap and 'delete' in tmp:
        tmp = tmp[~tmp.delete]

    if mobile:
        scale = width / 1000
    else:
        scale = 1

    tmp['font_size'] = np.ceil(scale * 13).astype(np.int8)
    if mobile and width < height:
        a, b = tmp['x'].copy(True), tmp['y'].copy(True)
        tmp['x'], tmp['y'] = b, a
        tmp['size'] = tmp['size'] * scale
        tmp['font_size'] = np.ceil(scale * 24).astype(np.int8)

    max_size = tmp['size'].max()

    sizeref = 2. * max_size / ((100 * scale) ** 2)
    # sizeref = max_size / 60 ** 2

    fig = go.Figure(data=[go.Scatter(
        x=tmp['x'],
        y=tmp['y'],
        mode='markers',
        text=tmp[text_col],
        fill=None,
        textposition=text_pos,
        marker_size=tmp['size'],
        marker=dict(
            sizemode='area',
            color="#ffffff",
            sizeref=sizeref,
            line_width=np.ceil(scale * 3).astype(np.int8),
            line_color=-tmp['size'],
            # colorscale='Viridis',
            # colorscale="pubugn"
        ),
        # textfont=dict(
        #     size=tmp['font_size'],
        # ),
        opacity=1,
        hovertemplate="%{text}<extra></extra>"
    )])

    for row in tmp.itertuples():
        fig.add_annotation(
            x=row.x,
            y=row.y,
            # xref="x",
            # yref="y",
            text=row.entity_s,
            showarrow=False,
            font=dict(
                # family="Courier New, monospace",
                size=row.font_size,
                # size=12,
                color="#636363"
            ),
            align="center",
            ax=20,
            ay=-30,
        )

    fig.update_layout(
        # title=title,
        # autosize=False,
        # width=700,
        height=height - 200,
        margin=dict(l=1, r=1, t=1, b=1),
        hoverlabel_align='right',
        # paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    fig.update_xaxes(showline=True, linecolor='#000000', linewidth=2)
    fig.update_yaxes(showline=True, linecolor='#000000', linewidth=2)
    if mobile:
        fig.update_layout(dragmode=False)
        fig.update_yaxes(showticklabels=False, automargin=True)
        fig.update_xaxes(showticklabels=True, automargin=True)
    # else:
    #     fig.update_xaxes(showline=True, linecolor='#000000', linewidth=2)
    #     fig.update_yaxes(showline=True, linecolor='#000000', linewidth=2)

    return fig.to_dict()


def generate_wordcloud_src(entities: pd.DataFrame):
    freq = multidict.MultiDict()
    for tup in entities.itertuples():
        freq.add(tup.entity, tup.salience)

    wc = WordCloud(background_color="white", max_words=1000, width=800, height=400)
    wc.generate_from_frequencies(freq)
    img = BytesIO()
    wc.to_image().save(img, 'PNG')

    return 'data:image/png;base64,{}'.format(base64.b64encode(img.getvalue()).decode())
