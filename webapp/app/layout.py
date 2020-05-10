import os

import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html


NAVBAR = dbc.Navbar(
    html.A(
        dbc.Row(
            [
                dbc.Col(html.Img(src=os.environ.get("LOGO"),
                                 height="50px")),
                dbc.Col(dbc.NavbarBrand("covid19 answers",
                                        className="ml-4",
                                        style={
                                            "color": "white",
                                            "font-size": "1.5rem"})),
            ],
            align="center",
            no_gutters=True,
        ),
        href="https://plot.ly",
    ),
    color="primary",
    sticky="top",
    expand="xs"
)

QUERY_INPUT = dbc.InputGroup(
    [
        dbc.Input(id="query-text", placeholder="Query ..."),
        dbc.InputGroupAddon(
            dbc.Button(
                html.Span([html.I(className="fas fa-search")]),
                color="success",
                id="query-button"),
            addon_type="append",
        ),
    ],
    style={"padding-top": "25px", "padding-bottom": "25px"}
)

LAYOUT = html.Div(
    [
        NAVBAR,
        dbc.Container([
            QUERY_INPUT,
            dbc.Spinner(html.Div(id="results"), color="success", type="grow")
        ])
    ]
)