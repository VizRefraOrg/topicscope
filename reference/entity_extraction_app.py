import warnings
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import base64
import hashlib
import re
import sys

import dash
import dash_bootstrap_components as dbc
import dash_html_components as html
import requests
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
from dash.exceptions import PreventUpdate
from requests.exceptions import HTTPError, ConnectionError

from service import pysummarizer, service
from apps import BaseApplication
from apps.entity_extraction.layouts import BODY, MAIN_TABS, UPLOAD_BAR, URL_BAR, SETTING_PANEL
from plot import graph
from service.service import *
from setting import API_KEY, API_URL

MAX_TEXT_SIZE = 30 * 1024
MAX_WORD = 100
MAX_LENGTH = 4000


def uuid(text):
    m = hashlib.md5()
    m.update(text.encode('utf-8'))
    return str(int(m.hexdigest(), 16))[12]


class EntityApplication(BaseApplication):

    def __init__(self):
        self.title = "VizRefra"
        BaseApplication.__init__(self)

    def create_layout(self):
        self.app.index_string = """<!DOCTYPE html>
            <html>
                <head>
                    <!-- Global site tag (gtag.js) - Google Analytics -->
                    <script async src="https://www.googletagmanager.com/gtag/js?id=G-9L0NEXXLJV"></script>
                    <script>
                      window.dataLayer = window.dataLayer || [];
                      function gtag(){dataLayer.push(arguments);}
                      gtag('js', new Date());
                    
                      gtag('config', 'G-9L0NEXXLJV');
                    </script>
                    {%metas%}
                    <title>{%title%}</title>
                    {%favicon%}
                    {%css%}
                </head>
                <body>
                    {%app_entry%}
                    <footer>
                        {%config%}
                        {%scripts%}
                        {%renderer%}
                    </footer>
                </body>
            </html>"""

        self.app.layout = html.Div(children=[
            BODY,
        ], style={'height': '100vh'})

        self.app.clientside_callback(
            """
            function(href) {
                var w = window.innerWidth;
                var h = window.innerHeight;
                var check = false;
                  (function(a){
                    if(/(android|bb\d+|meego).+mobile|avantgo|bada\/|blackberry|blazer|compal|elaine|fennec|hiptop|iemobile|ip(hone|od)|iris|kindle|lge |maemo|midp|mmp|mobile.+firefox|netfront|opera m(ob|in)i|palm( os)?|phone|p(ixi|re)\/|plucker|pocket|psp|series(4|6)0|symbian|treo|up\.(browser|link)|vodafone|wap|windows ce|xda|xiino/i.test(a)||/1207|6310|6590|3gso|4thp|50[1-6]i|770s|802s|a wa|abac|ac(er|oo|s\-)|ai(ko|rn)|al(av|ca|co)|amoi|an(ex|ny|yw)|aptu|ar(ch|go)|as(te|us)|attw|au(di|\-m|r |s )|avan|be(ck|ll|nq)|bi(lb|rd)|bl(ac|az)|br(e|v)w|bumb|bw\-(n|u)|c55\/|capi|ccwa|cdm\-|cell|chtm|cldc|cmd\-|co(mp|nd)|craw|da(it|ll|ng)|dbte|dc\-s|devi|dica|dmob|do(c|p)o|ds(12|\-d)|el(49|ai)|em(l2|ul)|er(ic|k0)|esl8|ez([4-7]0|os|wa|ze)|fetc|fly(\-|_)|g1 u|g560|gene|gf\-5|g\-mo|go(\.w|od)|gr(ad|un)|haie|hcit|hd\-(m|p|t)|hei\-|hi(pt|ta)|hp( i|ip)|hs\-c|ht(c(\-| |_|a|g|p|s|t)|tp)|hu(aw|tc)|i\-(20|go|ma)|i230|iac( |\-|\/)|ibro|idea|ig01|ikom|im1k|inno|ipaq|iris|ja(t|v)a|jbro|jemu|jigs|kddi|keji|kgt( |\/)|klon|kpt |kwc\-|kyo(c|k)|le(no|xi)|lg( g|\/(k|l|u)|50|54|\-[a-w])|libw|lynx|m1\-w|m3ga|m50\/|ma(te|ui|xo)|mc(01|21|ca)|m\-cr|me(rc|ri)|mi(o8|oa|ts)|mmef|mo(01|02|bi|de|do|t(\-| |o|v)|zz)|mt(50|p1|v )|mwbp|mywa|n10[0-2]|n20[2-3]|n30(0|2)|n50(0|2|5)|n7(0(0|1)|10)|ne((c|m)\-|on|tf|wf|wg|wt)|nok(6|i)|nzph|o2im|op(ti|wv)|oran|owg1|p800|pan(a|d|t)|pdxg|pg(13|\-([1-8]|c))|phil|pire|pl(ay|uc)|pn\-2|po(ck|rt|se)|prox|psio|pt\-g|qa\-a|qc(07|12|21|32|60|\-[2-7]|i\-)|qtek|r380|r600|raks|rim9|ro(ve|zo)|s55\/|sa(ge|ma|mm|ms|ny|va)|sc(01|h\-|oo|p\-)|sdk\/|se(c(\-|0|1)|47|mc|nd|ri)|sgh\-|shar|sie(\-|m)|sk\-0|sl(45|id)|sm(al|ar|b3|it|t5)|so(ft|ny)|sp(01|h\-|v\-|v )|sy(01|mb)|t2(18|50)|t6(00|10|18)|ta(gt|lk)|tcl\-|tdg\-|tel(i|m)|tim\-|t\-mo|to(pl|sh)|ts(70|m\-|m3|m5)|tx\-9|up(\.b|g1|si)|utst|v400|v750|veri|vi(rg|te)|vk(40|5[0-3]|\-v)|vm40|voda|vulc|vx(52|53|60|61|70|80|81|83|85|98)|w3c(\-| )|webc|whit|wi(g |nc|nw)|wmlb|wonu|x700|yas\-|your|zeto|zte\-/i.test(a.substr(0,4))) 
                      check = true;
                  })(navigator.userAgent||navigator.vendor||window.opera);

                return {'height': h, 'width': w, 'mobile': check};
            }
            """,
            Output('windowSize', 'data'),
            Input('url', 'href')
        )

        return self

    def add_callback(self):
        app = self.app

        @app.callback(
            [
                Output("text-bar", "style"),
                Output("url-bar", "style")
            ],
            [
                Input("select-feature", "value")
            ]
        )
        def select_feature(value):
            if value is None:
                raise PreventUpdate
            elif value == 'text':
                return {}, {'display': 'none'}
            elif value == 'url':
                return {'display': 'none'}, {}

        @app.callback(
            [
                Output("textarea", "value"),
                Output("fulltext-store", "data"),
                Output("auto-toast-big-text", "children"),
                Output("auto-toast-big-text", "is_open"),
                # Output("too-large-text", "color"),
                # Output("too-large-text", "duration")
            ],
            [
                Input('upload', 'contents'),
                State('fulltext-store', 'data'),
            ],
        )
        def upload_file(contents, data):
            info = None
            text = ""
            fulltext = ""

            if contents is None:
                raise PreventUpdate

            content_type, content_string = contents.split(',')
            if 'text' in content_type.lower():
                fulltext = base64.b64decode(content_string).decode('iso-8859-1').encode("ascii", "ignore").decode()
                fulltext = ' '.join([re.sub("[-_]+", "", line.strip()) for line in fulltext.splitlines() if line])

                size = sys.getsizeof(fulltext)

                if size > MAX_TEXT_SIZE:
                    # size bigger than 200kb
                    # truncate
                    tokens = fulltext.split()
                    limit = MAX_TEXT_SIZE * len(tokens) // size // 2
                    text = " ".join(tokens[:limit])
                    info = dbc.Alert(
                        "The input file is too big. It has been truncated!",
                        color="warning",
                    )
                else:
                    text = fulltext
            else:
                info = dbc.Alert(
                    "Only Support plain text file, Tap/click Try Again",
                    color="warning",
                )
            style = info is not None
            data = data or {}
            data[uuid(text)] = fulltext[:MAX_LENGTH]

            return text, data, info, style

        @app.callback(
            [
                Output("store", "data"),
                Output("graph-block", "style"),
                Output('auto-toast', 'is_open'),
                Output("auto-toast-error", "is_open"),
                Output("auto-toast-short-text", "is_open")
            ],
            [
                Input('submit-button', 'n_clicks'),
                Input('textarea', 'value'),
                Input('search-button', 'n_clicks'),
                Input('url-box', 'value'),
                State('textarea', 'value'),
                State('url-box', 'value'),
            ],
        )
        def submit_button(text_btt, text, url_btt, url, text_st, url_st):
            if text_btt is None and url_btt is None:
                # prevent the None callbacks is important with the store component.
                # you don't want to update the store for nothing.
                raise PreventUpdate

            changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
            if 'submit-button' in changed_id and text is not None:
                content = text
            elif 'search-button' in changed_id and url is not None:
                # scrape content
                content = service.extract_main_content(url)
            else:
                return {}, {"display": "none"}, False, False, False

            if content is None:
                return {}, {"display": "none"}, False, True, False

            # processing
            if len(content.strip()) <= MAX_WORD:
                return {}, {"display": "none"}, False, False, True

            payload = {
                "entity": {
                    "text": content,
                    "salience": 0.0019,
                    "limit": 20
                }
            }
            headers = {'Content-Type': 'application/json', 'token': API_KEY}

            try:
                response = requests.post(API_URL, headers=headers, json=payload)
                response.raise_for_status()
                json_data = response.json()
            except HTTPError as http_err:
                # print(f'HTTP error occurred: {http_err}')  # Python 3.6
                return {}, {"display": "none"}, False, True, False
            except (ValueError, ConnectionError) as e:
                # print(f'Other error occurred: {err}')  # Python 3.6
                return {}, {"display": "none"}, False, True, False

            # phi = pd.DataFrame(json_data.get('phi', []))
            # radius = pd.DataFrame(json_data.get('radius', []))

            circle = pd.json_normalize(json_data.get('circle', []))
            tags = json_data.get('tags', [])

            highlight = graph.highlighted_markup(content, tags) if tags else {}
            data = {}
            if not circle.empty:
                # circle['size'] = circle['size']
                df = add_remove_flag(circle, scale=0.06)
                data = df.to_dict('records')

            return {'data': data, 'highlight': highlight, 'text': content}, {}, True, False, False

        # @app.callback(
        #     [
        #         Output("fulltext-store", "data"),
        #     ],
        #     [
        #         Input('store', 'data'),
        #         State('fulltext-store', 'data'),
        #         State('store', 'modified_timestamp'),
        #     ],
        # )
        # def update_full_store(store_data, full_text_data, state2):
        #     if state2 is None:
        #         raise PreventUpdate
        #
        #     store_data = store_data or {}
        #     full_text_data = full_text_data or {}
        #
        #     fulltext = store_data.get("text")
        #     full_text_data[uuid(fulltext)] = fulltext[:MAX_LENGTH]
        #     return full_text_data

        # @app.callback(
        #     [Output('auto-toast', 'children'),
        #      Output('auto-toast', 'is_open')],
        #     [
        #         Input('submit-button', 'n_clicks'),
        #         Input('textarea', 'value')
        #     ]
        # )
        # def display_alert(n_clicks, value):
        #     changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
        #
        #     info = None
        #     if 'submit-button' in changed_id:
        #         if value and len(value.split()) > 100:
        #             info = dbc.Alert(
        #                 "Analysing the input text ...! Please wait util the graphs appear!",
        #                 color="info",
        #                 # duration=4000
        #             )
        #         elif value:
        #             info = dbc.Alert(
        #                 "Your input text is too short to recognise any entities. The graphs might not be displayed. Please enter better text ....",
        #                 color="warning",
        #                 # duration=4000
        #             )
        #     #style = {} if info is None else {'width': '89%', 'display': 'inline-block'}
        #     style = info is not None
        #     return info, style
        @app.callback(
            Output("kmean-store", "data"),
            [
                Input("store", "data"),
                Input('n_clusters', 'value'),
                Input('size_threshold', 'value'),
                Input('top_n', 'value'),
            ],
        )
        def apply_kmean(data, n_clusters, size_threshold, top_n):
            if data is None:
                raise PreventUpdate
            data = data or {}

            data_dict = data.get('data', [])
            if data_dict:
                df = pd.DataFrame(data_dict)

                clusters = run_k_mean(df, n_clusters, size_threshold, top_n)
                return {'data': clusters.to_dict('records')}
            return {}

        @app.callback(
            [
                Output("tsne-kmean", "figure"),
                Output("tsne-kmean", "style"),
                Output("tsne-kmean-popup", "figure"),
            ],
            [
                Input("kmean-store", "modified_timestamp"),
                State("kmean-store", "data"),
                State("windowSize", "data")
            ],
        )
        def display_kmean(chart_ts, chart_data, screen_data):
            if chart_ts is None:
                raise PreventUpdate

            screen_data = screen_data or {}
            chart_data = chart_data or {}
            data_dict = chart_data.get('data', [])
            if data_dict:
                df = pd.DataFrame(data_dict)
                figure = graph.create_2d_graph(df, text_col='entity', overlap=False, **screen_data)
                return figure, {}, figure
            return {}, {"display": "none"}, {}

        @app.callback(
            [
                Output("tsne-bar", "figure"),
                Output("tsne-bar", "style"),
                Output("tsne-bar-popup", "figure"),
            ],
            [
                Input("kmean-store", "data"),
            ],
        )
        def display_bar(data):
            data_dict = data.get('data', [])
            if data_dict:
                df = pd.DataFrame(data_dict)
                figure = graph.create_bar_graph(df, top=10)
                return figure, {}, figure
            return {}, {"display": "none"}, {}

        @app.callback(
            [
                Output("entity", "children"),
                Output("tsne-entity-popup", "children")
            ],
            [
                Input("store", "data"),
            ],
        )
        def display_entity(data):
            entity = data.get('highlight', [])
            if entity:
                return entity, entity
            return None, None

        @app.callback(
            [
                Output("tsne-3d", "figure"),
                Output("tsne-3d", "style"),
                Output("tsne-3d-popup", "figure")
            ],
            [
                Input("kmean-store", "data"),
                Input('xylim', 'value'),
                Input('gridN', 'value'),
                Input('spreadN', 'value'),
                State("kmean-store", "data"),
            ],
        )
        def display_3d(data, xylim, grid_n, spread_n, ts):
            if ts is None:
                raise PreventUpdate
            data = data or {}

            data_dict = data.get('data', [])
            if data_dict:
                df = pd.DataFrame(data_dict)

                grid_fig = graph.create_3d_graph(df, xylim, grid_n, spread_n)
                return grid_fig, {}, grid_fig
            return {}, {"display": "none"}, {}

        @app.callback(
            [
                Output("summarization", "children"),
                Output("tsne-summary-popup", "children"),
                Output("category", "children"),
                Output('sentiment-title', 'children'),
                Output("sentiment-chart", "children"),
                Output("wordcloud-chart", "children")
            ],
            [
                Input('store', 'modified_timestamp'),
                Input("fulltext-store", "modified_timestamp"),
                State('store', 'data'),
                State("fulltext-store", "data"),
            ],
        )
        def display_summary(ts1, ts2, data, full_data):
            if ts1 is None and ts2 is None:
                raise PreventUpdate

            data = data or {}
            text = data.get("text", "")
            full_data = full_data or {}

            fulltext = full_data.get(uuid(text), text)
            if len(fulltext.split()) > MAX_WORD:
                summary = ' '.join(pysummarizer.summarize(fulltext))
                entities, tags = service.google_entities(fulltext, limit=10, salience=0)

                highlight = html.P(summary)
                # highlight = graph.highlighted_markup(summary, tags)

                category, confidence = service.google_classify( fulltext )
                clazz = html.Mark(children=[
                    category,
                    html.Span(
                        f"{round(confidence*100, 3)}% confidence",
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
                ], style={"backgroundColor": "lightgrey", "color": "black"})

                flag, score, sentiments = service.google_sentiment( fulltext )

                #sentiment_title = html.B(f"{flag} sentiment with overall score: <i>{round(score * 100, 3)}%</i>")
                sentiment_title = html.Mark(children=[
                    f"{flag}",
                    html.Span(
                        f"{round(score * 100, 3)}% score",
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
                ], style={"backgroundColor": "lightgrey", "color": "black"})

                sentiment_fig_src = graph.create_pie(values=list(sentiments.values()), names=list(sentiments.keys()))
                sentiment_fig = html.Img(id="pie-img", src=sentiment_fig_src, style={'width': '100%'})
                wordcloud = html.Img(id="wc-img", src=graph.generate_wordcloud_src(entities))

                return highlight, highlight, [html.B('Classification: '), clazz], [html.B("Sentiment: "), sentiment_title], [sentiment_fig], [wordcloud]
            return None, None, None, [html.P("The input text is too short to understand! Please enter a longer text!")], None, None

        @app.callback(
            Output("collapse", "is_open"),
            [
                Input("collapse-button", "n_clicks"),
                Input("tsne-kmean", "style"),
                Input("select-feature", "value"),
                # Input('textarea', 'value'),
                # Input('submit-button', 'n_clicks')
            ],
            [State("collapse", "is_open")],
            prevent_initial_call=True
        )
        def toggle_collapse(n, figure_style, feature, is_open):
            changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
            if 'collapse-button' in changed_id or len(figure_style) == 0:
                return not is_open

            elif not is_open and feature in ['text', 'url']:
                print(is_open, feature)
                return not is_open

            return is_open

        @app.callback(
            Output("modal-centered", "is_open"),
            [
                Input("popup-button", "n_clicks"),
            ],
            [State("modal-centered", "is_open")],
            prevent_initial_call=True
        )
        def toggle_modal(n1, is_open):
            if n1:
                return not is_open
            return is_open

        @app.callback(
            Output("modal-bar", "is_open"),
            [
                Input("popup-bar", "n_clicks"),
            ],
            [State("modal-bar", "is_open")],
            prevent_initial_call=True
        )
        def toggle_bar(n1, is_open):
            if n1:
                return not is_open
            return is_open

        @app.callback(
            Output("modal-centered2", "is_open"),
            [
                Input("popup-er", "n_clicks"),
            ],
            [State("modal-centered2", "is_open")],
            prevent_initial_call=True
        )
        def toggle_modal2(n1, is_open):
            if n1:
                return not is_open
            return is_open

        @app.callback(
            Output("modal-centered3", "is_open"),
            [
                Input("popup-3d", "n_clicks"),
            ],
            [State("modal-centered3", "is_open")],
            prevent_initial_call=True
        )
        def toggle_modal3(n1, is_open):
            if n1:
                return not is_open
            return is_open

        @app.callback(
            Output("modal-centered4", "is_open"),
            [
                Input("popup-sum", "n_clicks"),
            ],
            [State("modal-centered4", "is_open")],
            prevent_initial_call=True
        )
        def toggle_modal3(n1, is_open):
            if n1:
                return not is_open
            return is_open
