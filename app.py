import json
from astropy.utils.misc import indent
from networkx.algorithms import components
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import odakb
import rdflib
import time

import requests, os

from copy import deepcopy
#import base64
import integralclient as ic

from astropy.time import Time, TimeDelta
from astropy import units as u
import io
from astropy.io import fits


# Use the non-interactive Agg backend, which is recommended as a
# thread-safe backend.
# See https://matplotlib.org/3.3.2/faq/howto_faq.html#working-with-threads.
import matplotlib as mpl
from streamlit.proto.Markdown_pb2 import Markdown
from traitlets.traitlets import default
mpl.use("agg")

import logging
logging.getLogger("oda.kb.sparql").setLevel('DEBUG')


##############################################################################
# Workaround for the limited multi-threading support in matplotlib.
# Per the docs, we will avoid using `matplotlib.pyplot` for figures:
# https://matplotlib.org/3.3.2/faq/howto_faq.html#how-to-use-matplotlib-in-a-web-application-server.
# Moreover, we will guard all operations on the figure instances by the
# class-level lock in the Agg backend.
##############################################################################
from matplotlib.backends.backend_agg import RendererAgg
_lock = RendererAgg.lock


# -- Set page config
apptitle = 'ODA Quickview'

st.set_page_config(page_title=apptitle, page_icon=":eyeglasses:", layout="wide")
# -- Default detector list
#detectorlist = ['H1','L1', 'V1']

# Title the app

paper_ns = rdflib.Namespace('http://odahub.io/ontology/paper#')
topic_ns =  rdflib.Namespace('http://odahub.io/ontology/paper/topic#')
astroobject_ns = rdflib.Namespace('http://odahub.io/ontology/astroobject#')


st.title('MMODA Source Quick-Look: INTEGRAL (SPI-ACS + ISGRI), POLAR')


st.write(f"""<div width="100%" align="right"> <a class="logo navbar-btn pull-left"
        target="_blank"
        href="https://www.astro.unige.ch/mmoda/"
        title="Home"> <img
        height="50px"
        src="https://www.astro.unige.ch/mmoda/sites/all/themes/bootstrap_astrooda/logo.png" alt="Home" />
      </a>
      <a class="logo navbar-btn pull-left" target="_blank"
        target="_blank"
        href="https://www.unige.ch/sciences/astro/en/"
        title="Departement of Astronomy - university of Geneva"> <img
        height="50px"
        src="https://www.astro.unige.ch/mmoda/sites/all/themes/bootstrap_astrooda/logo-fac-sciences.png" alt="Departement of Astronomy - university of Geneva" />
      </a>
      <a class="logo navbar-btn pull-left" target="_blank"
        target="_blank"
        href="https://www.isdc.unige.ch/integral/"
        title="The INTErnational Gamma-Ray Astrophysics Laboratory - INTEGRAL"> <img
        height="50px"
        src="https://www.astro.unige.ch/mmoda/sites/all/themes/bootstrap_astrooda/logo-isdc.png" alt="The INTErnational Gamma-Ray Astrophysics Laboratory - INTEGRAL" />
      </a>
      <a class="logo navbar-btn pull-left"
        target="_blank"
        href="https://www.epfl.ch/labs/lastro" target="_blank"
        title="Laboratory of Astrophysics (LASTRO) - EPFL"> <img
        height="50px"
        src="https://www.astro.unige.ch/mmoda/sites/all/themes/bootstrap_astrooda/logo-epfl.png" alt="Laboratory of Astrophysics (LASTRO) - EPFL" />
      </a>
      <a class="logo navbar-btn pull-left"
        target="_blank"
        href="https://apc.u-paris.fr/APC_CS/en" target="_blank"
        title="Laboratoire AstroParticule et Cosmologie (APC)"> <img
        height="50px"
        src="https://www.astro.unige.ch/mmoda/sites/all/themes/bootstrap_astrooda/logo-apc.png" alt="Laboratoire AstroParticule et Cosmologie (APC)" />
      </a>
      <br>
      <a class="logo navbar-btn pull-left"
        target="_blank"
        href="http://doi.org/10.17616/R38P6F" target="_blank"
        title="INTEGRAL Archive"> <img
        height="50px"
        src="https://www.re3data.org/public/badges/s/light/100010657.svg" alt="Laboratoire AstroParticule et Cosmologie (APC)" />
      </a>
      </div>
      """, unsafe_allow_html=True)


#TODO: sparql and rdf!

st.markdown("""
 * All data fetched from https://www.astro.unige.ch/mmoda/, go there to access everything!
 * Use the menu at left to select data and set plot parameters. Your plots will appear below
 * *Link with out graph!* Linked Open Data: [SPARQL](), current [RDF]().
****
""")


st.markdown("""
## Linked Open Data
 * *Link with out graph!* Linked Open Data: [SPARQL](), current [RDF]().
 * Data sources, events, workflows
 * See associated [project](https://github.com/oda-hub/renku-aqs/tree/upgraph)
****
""")

import oda_api.api

from astropy.time import Time



@st.cache(ttl=3600, max_entries=10, persist=True)   #-- Magic command to cache data
def load_spiacs_lc(t0, dt_s):
    _t0 = Time(t0, format="isot")

    disp = oda_api.api.DispatcherAPI(url="https://www.astro.unige.ch/mmoda/dispatch-data/")

    t1_isot = Time(_t0.mjd - dt_s/24/3600, format="mjd").isot
    t2_isot = Time(_t0.mjd + dt_s/24/3600, format="mjd").isot

    print("t1, t2", t1_isot, t2_isot)

    try:
        d = disp.get_product(
            instrument="spi_acs",
            product="spi_acs_lc",
            time_bin=0.1,
            T_format="isot",
            T1=t1_isot,
            T2=t2_isot,
            product_type='Real',
        )
        lc = d.spi_acs_lc_0_query.data_unit[1].data

        print("lc", str(lc)[:300])
    except oda_api.api.RemoteException:
        return

    return lc



@st.cache(ttl=3600, max_entries=10, persist=True)   #-- Magic command to cache data
def load_polar_lc(t0, dt_s):
    _t0 = Time(t0, format="isot")

    disp = oda_api.api.DispatcherAPI(url="https://www.astro.unige.ch/mmoda/dispatch-data/")

    _t1 = Time(_t0.mjd - dt_s/24/3600, format="mjd")
    _t2 = Time(_t0.mjd + dt_s/24/3600, format="mjd")

    t1_isot = _t1.isot    
    t2_isot = _t2.isot

    if _t1 < Time("2016-08-01T00:00:00", format="isot") or _t2 > Time("2017-05-01T00:00:00", format="isot"):
        raise RuntimeError("No POLAR data")

    print("t1, t2", t1_isot, t2_isot)

    d = disp.get_product(
        instrument="polar",
        product="polar_lc",
        time_bin=0.1,
        T_format="isot",
        T1=t1_isot,
        T2=t2_isot,
        product_type='Real',
    )

    print(dir(d))

    lc = d.polar_lc_0_lc.data_unit[1].data

    print("lc", lc)
    
    return lc


@st.cache(ttl=300, max_entries=100, persist=True)   #-- Magic command to cache data
def load_event_papers(name):
    try:
        import odakb
        D = odakb.sparql.select(
            f'?paper ?x "{name}"; ?p ?o', 
            '?paper ?p ?o',
            tojdict=True,
            limit=3000)        
        D.update(odakb.sparql.select(
            f'?paper paper:mentions_named_grb "{name}"; ?p ?o', 
            '?paper ?p ?o',
            tojdict=True,
            limit=3000))

        D.update(odakb.sparql.select(
            f'?paper paper:mentions_named_event "{name}"; ?p ?o', 
            '?paper ?p ?o',
            tojdict=True,
            limit=3000))

        print("D:", D)

        return D
                #jq -cr '.[] | .["http://odahub.io/ontology/paper#grb_isot"][0]["@value"] + "/" + .["http://odahub.io/ontology/paper#mentions_named_grb"][0]["@value"]' | \
                #sort -r | head -n${nrecent:-20}
    except Exception as e:
        raise RuntimeError("PROBLEM listing GRBs:", e)

@st.cache(ttl=300, max_entries=100, persist=True)   #-- Magic command to cache data
def load_objects_of_interest():
    try:
        import odakb
        D = odakb.sparql.select(
            '''
            ?object an:name ?object_name; 
                    an:importantIn ?domain;
                    ?p ?o .
            ''',          
            '?object ?p ?o',  
            tojdict=True,
            limit=100)        

        print("D:", D)

        return D
                #jq -cr '.[] | .["http://odahub.io/ontology/paper#grb_isot"][0]["@value"] + "/" + .["http://odahub.io/ontology/paper#mentions_named_grb"][0]["@value"]' | \
                #sort -r | head -n${nrecent:-20}
    except Exception as e:
        raise
        raise RuntimeError("PROBLEM listing objects of interst:", e)

objects_of_interest = load_objects_of_interest()
    

    
@st.cache(ttl=600, allow_output_mutation=True)
def load_events(kind="grb", recent_paper_days=30*6, with_details=True, time_seq=0):
    try:
        t0 = time.time()

        D = odakb.sparql.query(
            f'''
            PREFIX paper: <http://odahub.io/ontology/paper#>

            DESCRIBE ?paper WHERE {{
                ?paper paper:timestamp ?timestamp .
                
                FILTER (
                    ?timestamp > {time.time() - recent_paper_days*24*3600}
                )
            }}
            
            LIMIT 100000''', 
            )

        # print("GT", D)
        
            
        try:
            D = D['problem-decoding'] #['results']['bindings']
        except: 
            pass


        G = rdflib.Graph()

        try:
            G.parse(data=D, format='turtle')
        except Exception as e:
            print("problem:", D)
            raise

        print(f'for last {recent_paper_days} days got {len(G)} entries in {time.time() - t0} s')
        
        D = G.query(f'''
            PREFIX paper: <http://odahub.io/ontology/paper#>

            SELECT DISTINCT ?name WHERE {{
                ?paper paper:mentions_named_{kind}|paper:reports_event ?name;
                       paper:timestamp ?timestamp .

            }}
            

            ORDER BY DESC(?isot)
            ''')


        D = G.query(f'''
            PREFIX paper: <http://odahub.io/ontology/paper#>

            SELECT ?paper ?timestamp ?url ?name ?isot ?ra ?dec ?citation ?topic WHERE {{
                ?paper paper:mentions_named_{kind}|paper:reports_event ?name;
                       paper:timestamp ?timestamp .

                OPTIONAL {{
                    ?paper paper:url|paper:location ?url .
                }}

                OPTIONAL {{
                    ?paper paper:cites ?citation .
                }}

                OPTIONAL {{
                    ?paper paper:topics ?topic .
                }}


                { "OPTIONAL {{" if not with_details else "" }
                    ?paper paper:grb_isot|paper:event_isot ?isot;
                        paper:gbm_ra|paper:event_ra ?ra;
                        paper:gbm_dec|paper:event_dec ?dec;
                { "}}" if not with_details else "" }
      
                FILTER (
                    ?timestamp > {time.time() - recent_paper_days*24*3600}
                )

            }}
            

            ORDER BY DESC(?isot)
            ''')

        result = {}

        t0 = time.time()

        for paper, timestamp, url, name, isot, ra, dec, citation, topic in D:
            if not isinstance(name, rdflib.Literal):
                # print(">>>", name)
                continue


            # print(name, paper)

            paper = {paper: url}
            
            if name in result:
                p = {**result[name]['papers'], **paper }
            else:
                p = paper

            result[name] = {
                'isot': isot,
                'ra': ra,
                'dec': dec,
                'timestamp': timestamp,
                'papers': p
            }

        print(f"sub-selection in {time.time() - t0}")

        for paper, event, characteristic in G.query(f'''
            SELECT ?paper ?event ?characteristic WHERE {{
                ?paper paper:mentions_named_event ?event;
                       paper:reports_characteristic ?characteristic .                       
            }}
        '''):
            # print("!", paper, event, characteristic)
            G.add((paper, paper_ns['reports_' + characteristic + '_for'],  astroobject_ns[event]))

        for paper, _p, event in G.triples((None, paper_ns['mentions_named_event'], None)):
            G.add((paper, _p, astroobject_ns[event]))
            #G.add((paper_ns[event], ))
        
        return result, G

    except Exception as e:
        raise


@st.cache(ttl=3600, max_entries=10, persist=True)   #-- Magic command to cache data
def load_integral_time(t0):
    _t0 = Time(t0, format="isot")
    d = ic.converttime("UTC", _t0.isot, "ANY")
    return d


@st.cache(ttl=3600, max_entries=10, persist=True)   #-- Magic command to cache data
def load_fermi_time(t0):
    import requests
    import re

    xtime_url="https://heasarc.gsfc.nasa.gov/cgi-bin/Tools/xTime/xTime.pl"
    pattern='<tr>(.*?)</tr>'
    #sub_pattern='.*?<th scope=row><label for="(.*?)">.*?</label></th>.*?<td align=center>.*?</td>.*?<td>(.*?)</td>.*?'
    sub_pattern='<th scope=row><label for="(.*?)">.*?</label></th>.*?<td align=center>.*?</td>.*?<td.*?>(.*?)</td>'
    #pattern='<tr>.*?<th scope=row><label for="(.*?)">.*?</label></th>.*?<td align=center>.*?</td>.*?<td>(.*?)</td>.*?</tr>'

    def queryxtime(**args):
        args={**args, 
              **dict(
                    timesys_in="u",
                    timesys_out="u",
                    apply_clock_offset="yes")
                }

        content=requests.get(xtime_url,params=args).text

        #print("content",content)

        r=[]
        
        for tr in re.findall(pattern,content,re.S):
            print("tr",tr)
            s = dict(re.findall(sub_pattern,tr,re.S))
            print("s",s)

            r+=list(s.items())

        return dict(r)

    return queryxtime(time_in_i=t0)

@st.cache(ttl=3600, max_entries=10, persist=True)   #-- Magic command to cache data
def load_integral_sc(t0, ra_deg, dec_deg):
    _t0 = Time(t0, format="isot")
    if ra_deg is None or dec_deg is None:
        d = ic.get_sc(_t0.isot)
    else:
        d = ic.get_sc(_t0.isot, ra=ra_deg, dec=dec_deg)
    return d


@st.cache(ttl=3600, max_entries=10, persist=True)   #-- Magic command to cache data
def load_isgri(t0, dt_s):
    _t0 = Time(t0, format="isot")

    disp = oda_api.api.DispatcherAPI(url="https://www.astro.unige.ch/mmoda/dispatch-data/")

    t1_isot = Time(_t0.mjd - dt_s/24/3600, format="mjd").isot
    t2_isot = Time(_t0.mjd + dt_s/24/3600, format="mjd").isot

    print("t1, t2", t1_isot, t2_isot)

    import integralclient as ic

    d = ic.converttime("UTC", _t0.isot, "ANY")

    print(d)

    scw = d['SCWID']

    print("scw:", scw)

    url = f"http://isdcarc.unige.ch//arc/rev_3/scw/{scw[:4]}/{scw}.001/isgri_events.fits.gz"

    print("url", url)

    c = requests.get(url).content 
    # f = io.BytesIO(
    #     #requests.get(f"https://isdcarc.unige.ch/arc/FTP/arc_distr/NRT/public/scw/{scw[:4]}/{scw}/isgri_events.fits.gz").content
    #     c
    # )
    # f.seek(0)

    with open("tmp.fits.gz", "wb") as _:
        _.write(c)
    
    f = fits.open("tmp.fits.gz")
    d = f[3].data

    return d



@st.cache(ttl=3600, max_entries=10, persist=True)   #-- Magic command to cache data
def load_spi(t0, dt_s):
    _t0 = Time(t0, format="isot")

    disp = oda_api.api.DispatcherAPI(url="https://www.astro.unige.ch/mmoda/dispatch-data/")

    t1_isot = Time(_t0.mjd - dt_s/24/3600, format="mjd").isot
    t2_isot = Time(_t0.mjd + dt_s/24/3600, format="mjd").isot

    print("t1, t2", t1_isot, t2_isot)

    import integralclient as ic

    d = ic.converttime("UTC", _t0.isot, "ANY")

    print(d)

    scw = d['SCWID']

    print("scw:", scw)

    url = f"http://isdcarc.unige.ch//arc/rev_3/scw/{scw[:4]}/{scw}.001/spi_oper.fits.gz"

    print("url", url)

    c = requests.get(url).content 
    # f = io.BytesIO(
    #     #requests.get(f"https://isdcarc.unige.ch/arc/FTP/arc_distr/NRT/public/scw/{scw[:4]}/{scw}/isgri_events.fits.gz").content
    #     c
    # )
    # f.seek(0)

    with open("tmp.fits.gz", "wb") as _:
        _.write(c)
    
    f = fits.open("tmp.fits.gz")

    print(f)

    for e in f:
        print(e.header.get('EXTNAME'))

    d = f['SPI.-OSGL-ALL'].data

    print(d.columns)

    return d

@st.cache(ttl=3600, max_entries=10, persist=True)   #-- Magic command to cache data
def load_isgri_image(t0):
    _t0 = Time(t0, format="isot")

    d = ic.converttime("UTC", _t0.isot, "ANY")

    print(d)

    scw = d['SCWID']

    print("scw:", scw)

    disp = oda_api.api.DispatcherAPI(url="https://www.astro.unige.ch/mmoda/dispatch-data/")
    ima = disp.get_product(
        instrument="isgri", 
        product="isgri_image", 
        product_type='Real',
        osa_version="OSA10.2",
        scw_list=[f"{scw}.001"]
    )

    return ima.mosaic_image_0_mosaic.data_unit[4].data

import io

@st.cache(ttl=10, max_entries=10, persist=True)   #-- Magic command to cache data
def load_ibis_veto(t0, dt_s):
    t = requests.get(f"https://www.astro.unige.ch/cdci/astrooda/dispatch-data/gw/integralhk/api/v1.0/genlc/IBIS_VETO/{t0}/{dt_s}").json()
    print("\033[31m", t, "\033[0m")
    d = np.genfromtxt(io.StringIO(t), names=['ijd', 'time', 'rate', 'x'])
    return d

# -- Get list of events
# find_datasets(catalog='GWTC-1-confident',type='events')
# eventlist = datasets.find_datasets(type='events')
# eventlist = [name.split('-')[0] for name in eventlist if name[0:2] == 'GW']
# eventset = set([name for name in eventlist])
# eventlist = list(eventset)
# eventlist.sort()

st.markdown("***")

st.write("### Last days in the sky:")

# with st.expander("More"):
n_last_days = st.slider("", 1, 30, 3)


now_in_the_sky, G = load_events("event", recent_paper_days=n_last_days, with_details=False)

# rdf2dot all the way down!
from rdf2dot import rdf2dot

no_text_G = rdflib.Graph()
#no_text_G.bind('paper', dict(G.namespaces())['paper'])
no_text_G.bind('.', dict(G.namespaces())['paper'])

for a, b, c in G:
    print("will draw triple >>>", a, b, c)
    if 'afterglow' in b:
        b = paper_ns["afterglow"]
    if 'topic' in b:
        b = paper_ns["topic"]
        c = topic_ns[c]
    elif 'cites' in b:
        b = paper_ns["cites"]
        if str(c).startswith('http'):
            c = rdflib.URIRef(c)
    else:
        b = paper_ns["m"]            
    
    if isinstance(c, rdflib.Literal):
        pass
    else:
        no_text_G.add((a, b, c))
        # print(">>", a, b, c)

open("these_last_days.ttl", "w").write(G.serialize(format='turtle'))



s = ""
for k, v in sorted(now_in_the_sky.items(), key=lambda a: -float(a[1]['timestamp'])):
    s += f"""<span class='highlight-small green'>
                        <a class='block' href="?source_name={k}">{k}</a>
                </span> &nbsp;"""
    
    s += "("
    s += ", ".join([f"<a href='{l}'  target='_blank'>{i.split('#')[1]}</a>" for i, l in v['papers'].items()])
    s += ")"
    
    s +=  "&nbsp;"*5

st.write(s, unsafe_allow_html=True)

# with st.expander("More"):
    
vis_width = 1200
vis_height = 800

fn = "these_last_days.ttl"
with open(fn) as f:
    st.markdown("")
    st.download_button('Download RDF/Turtle', f, file_name=fn)

drawer = st.radio('', ('pyvis', 'dot/circo'))

if drawer == 'dot/circo':
    dot_fn = 'g.dot'
    rdf2dot.rdf2dot(no_text_G, open(dot_fn, "w"))
    try:            
        os.system(f'< {dot_fn} circo -Tpng -oa.png')
        st.image(open('a.png', 'rb').read())
    except:
        st.markdown('no graphvis, can not!')
else:
    import pyvis
    import networkx

    dot_fn = 'g_no_html.dot'
    rdf2dot.rdf2dot(no_text_G, open(dot_fn, "w"), html_labels=False)        

    nx = networkx.drawing.nx_pydot.read_dot(dot_fn)

    g = pyvis.network.Network(height=f'{vis_height}px', width=f'{vis_width}px')
    g.repulsion(node_distance=150, central_gravity=0.33,
                spring_length=110, spring_strength=0.10,
                damping=0.95)

    g.from_nx(nx)
    #g.t

    my_html_fn = 'ex.html'
    g.save_graph(my_html_fn)

    import re

    #https://visjs.github.io/vis-network/examples/network/events/interactionEvents.html

    extra_js = """
        makeId = () => {
            let ID = "";
            let characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
            for ( var i = 0; i < 12; i++ ) {
                ID += characters.charAt(Math.floor(Math.random() * 36));
            }
            return ID;
        }

        network.on( 'click', function(properties) {
            var ids = properties.nodes;
            var clickedNodes = nodes.get(ids);
            console.log('clicked nodes:', clickedNodes);
            clicked_id = clickedNodes[0].id
            new_node_id = makeId();
            nodes.add({ id: new_node_id, label: clickedNodes[0].label});
            
            network.body.data.edges.add([{from: new_node_id, to: clicked_id}])
            
        });

        network.on( 'showPopup', function(properties) {
            var ids = properties.nodes;
            var cNodes = nodes.get(ids);
            console.log('showPopup nodes:', cNodes);
        });

        network.on( 'blurNode', function(properties) {
            var ids = properties.nodes;
            var cNodes = nodes.get(ids);
            console.log('blur nodes:', cNodes);
        });
        

        network.on( 'doubleClick', function(properties) {
            var ids = properties.nodes;
            var clickedNodes = nodes.get(ids);
            console.log('clicked nodes:', clickedNodes);
            clicked_id = clickedNodes[0].id
            new_node_id = makeId();
            nodes.add({ id: new_node_id, label: clickedNodes[0].label});                        
        });        
    """

    n = re.sub('</body>', f'<script>{extra_js}</script></body>', open(my_html_fn).read())
    open(my_html_fn, "w").write(n)

    st.components.v1.html(open(my_html_fn).read(), width=vis_width, height=vis_height)


st.markdown("***")


st.sidebar.markdown("# Select Astrophysical Source")

use_kg_grb = st.sidebar.checkbox('Load KG GRBs from GCNs', value=True)
use_kg_ooi = st.sidebar.checkbox('Load KG Objects of Interest of INTEGRAL QLA', value=True)

time_seq = int(time.time())%60

if use_kg_grb:
    try:
        kg_grb_list, D = load_events("grb", time_seq=time_seq)
        st.markdown(f'Loaded {len(kg_grb_list)} transients from KG, the last one is {list(sorted(kg_grb_list.keys()))[-1]}!')
    except Exception as e:
        raise
        st.markdown(f'sorry, could not load GRB list from KG. Maybe try later. Sorry.')
        kg_grb_list = {}

else:
    kg_grb_list = {}

eventlist = {
    "GRB170817A": '2017-08-17T12:41:00',
    "GRB080319B": '2008-03-19T06:12:44',
    "GRB120711A": "2012-07-11T02:45:30.0",
    "GRB190114C": "2019-01-14T20:57:02.38",
    **{k: d['isot'] for k, d in kg_grb_list.items()}
}
    
source_list = list(eventlist.keys())

if use_kg_ooi:
    source_list += list([v['an:name'][0] for k, v in objects_of_interest.items()])

url_source_names = st.experimental_get_query_params().get('source_name', [])

if url_source_names == []:
    url_source_name = None
else:
    url_source_name = url_source_names[0]

source_list += url_source_names

def sort_key(n):
    if n in eventlist and eventlist[n] is not None:
        return "z" + str(eventlist[n])

    return n

source_list = list(reversed(sorted(set(source_list), key=sort_key)))

if url_source_name is not None:
    custom_source = st.sidebar.selectbox('Custom name or known to us?', ['Custom', 'Known']) == 'Custom'
else:
    custom_source = st.sidebar.selectbox('Custom name or known to us?', ['Known', 'Custom']) == 'Custom'
                                    
if not custom_source:
    #use_kg_grb = st.sidebar.checkbox('Load KG GRBs from GCNs', value=True)    

    source_name = st.sidebar.selectbox('Select Source', source_list)
    #st.experimental_set_query_params(source_name=source_name)
else:
    source_name = st.sidebar.text_input('Select Source', url_source_name)
    

st.sidebar.markdown("## Select Observation Time")

query_t0_utc = st.experimental_get_query_params().get('t0_utc', [])

if query_t0_utc != []:
    query_t0_utc = query_t0_utc[0]

    #-- Set time by GPS or event
    select_event = st.sidebar.selectbox('How do you want to select time frame?',
                                        ['By UTC', 'By event name'])
else:
    select_event = st.sidebar.selectbox('How do you want to select time frame?',
                                        ['By event name', 'By UTC'])


if select_event == 'By UTC':
    # -- Set a GPS time:        

    if query_t0_utc != []:
        t0 = st.sidebar.text_input('UTC', query_t0_utc).strip()    # -- GW150914
    else:
        t0 = st.sidebar.text_input('UTC', '2017-01-05T06:14:07').strip()    # -- GW150914

else:
    #t0 = st.sidebar.text_input('UTC', '2008-03-19T06:12:44')    # -- GW150914

    
    t0 = eventlist.get(source_name, None)

    try:
        grb_papers = load_event_papers(source_name)
    except:
        grb_papers = {}

    #st.write(str(grb_papers))

    st.subheader(source_name)

    if kg_grb_list != {}:
        #D = kg_grb_list.get(chosen_event)
        
                
        
        for paper in reversed(sorted(grb_papers.values(), key=lambda x: x.get('paper:DATE',[''])[0])):
            cols = st.columns(3)
            #cols[0].write(paper.keys())
            cols[0].write(f"[{paper.get('paper:NUMBER', [''])[0]}]({paper.get('paper:location', [''])[0]}) {paper.get('paper:DATE', [''])[0]}")            
            cols[1].write(paper.get('paper:title', [''])[0])
            #cols[3].write(paper.keys())
            cols[2].write(paper.get('paper:gcn_authors', [''])[0][:100]+ "...")
        # for k, v in D.items():
        #     cols[0].write(k)
        #     cols[1].write(v[0])

#        st.dataframe(pd.DataFrame(dicts))

class Secret(object):
    @property
    def secret_location(self):
        if 'CDCI_SECRET_LOCATION' in os.environ:
            return os.environ['CDCI_SECRET_LOCATION']
        else:
            return os.environ['HOME']+"/.secret-cdci"

    def get_auth(self):
        username="cdci" # keep separate from secrect to cause extra confusion!
        password=os.environ.get("CDCI_SECRET",None)
        if password is None:
            password = open(self.secret_location).read()
        password=password.strip()
        return requests.auth.HTTPBasicAuth(username, password)

try:
    auth=Secret().get_auth()
except:
    auth=None

if t0 is None:
    t0 = ":".join(Time.now().isot.split(":")[:-2] + ["00","00"])

from load_css import local_css

local_css("style.css")
 
st.write(f"<span class='highlight red'>T<sub>{0}</sub> = {t0}</span>", unsafe_allow_html=True)
        
#use_gbm = st.sidebar.checkbox('Load GBM')
use_gbm = False


if auth is not None:
    use_ias = st.sidebar.checkbox('Load All Sky Rate Search')    

    if use_ias:
        @st.cache(ttl=60, max_entries=10, persist=False)   #-- Magic command to cache data
        def load_integral_all_sky(t0):
            url = f"https://oda-workflows-integral-all-sky.odahub.io/api/v1.0/get/integralallsky?t0_utc={t0}&_async_request=yes"
            r = requests.get(url, auth=auth)    
            return r.json()


        integral_all_sky = load_integral_all_sky(t0)

        json.dump(integral_all_sky, open("ias.json", "w"))

        st.markdown(integral_all_sky['workflow_status'])

        if integral_all_sky['workflow_status'] == "done":
            st.markdown(integral_all_sky.keys())

            import base64

            st.image(base64.b64decode(integral_all_sky['data']['output']['acs_lc_png_content']))

            st.image(base64.b64decode(integral_all_sky['data']['output']['excesses_mosaic_png_content']))

        


#-- Choose detector as H1, L1, or V1
#detector = st.sidebar.selectbox('Detector', detectorlist)

if t0 is not None:
    st.sidebar.markdown('## Set Plot Parameters')
    dtboth = st.sidebar.slider('Time Range (seconds)', 0.5, 1000.0, 50.0)  # min, max, default

    dt_rebin = st.sidebar.slider('Rebinning time scale (seconds)', max(0.1, dtboth/100), min(100.0, dtboth/10), dtboth/30)  # min, max, default

    isgri_e1 = st.sidebar.slider('ISGRI E_MIN (keV)', 15, 800, 25)
    isgri_e2 = st.sidebar.slider('ISGRI E_MAX (keV)', isgri_e1, 800, isgri_e1 + 100)

scope_d = st.sidebar.slider('Exploration scope (days)', 0.1, 100.0, 20.0)  # min, max, default


st.sidebar.markdown("## Select Sky Location")

select_loc = st.sidebar.selectbox('Sky Location',
                                ['By name', 'RA and Dec'])


from astropy import coordinates as coords
from astroquery.simbad import Simbad


if select_loc == "By name":
    source_coord = None

    try:
        source_coord = coords.SkyCoord(
            kg_grb_list[source_name]['ra'],
            kg_grb_list[source_name]['dec'],
            unit="deg"
        )
    except (KeyError, TypeError):
    
        try:
            t = Simbad.query_object(source_name)
        except Exception as e:
            t = None

        source_coord = None
        if t is not None:
            try:
                source_coord = coords.SkyCoord(
                    t['RA'][0],
                    t['DEC'][0],
                    unit=("hourangle", "deg")
                )
            except ValueError:
                pass

#    st.markdown("source:" + str(source_coord))
else:
    source_coord = coords.SkyCoord(
        st.sidebar.text_input('RA', '83').strip(),
        st.sidebar.text_input('Dec', '22').strip(),
        unit="deg"
    )

st.write(f"<span class='highlight blue'>{source_coord}</span>", unsafe_allow_html=True)


if t0 is not None:
    dt_s = dtboth / 2.0
    dt_s_download = (int(dt_s/100)+1)*100

    t0_xtime = load_fermi_time(t0)

# st.markdown(f"""
# {t0_xtime}
# """)

integral_time = load_integral_time(t0)

if source_coord is None:
    integral_sc = load_integral_sc(t0, None, None)
else:
    integral_sc = load_integral_sc(t0, source_coord.ra.deg, source_coord.dec.deg)



st.markdown(f"""
## INTEGRAL spacecraft at selected time
***
""")

col1, col2, col3 = st.columns(3)

try:
    with col1:
        st.markdown(f"""
        INTEGRAL ScW: {integral_time['SCWID']}

        {integral_sc['bodies']['earth']['separation']} km from Earth
        Pointing to RA={integral_sc['scx']['ra']}, Dec={integral_sc['scx']['dec']}

        """)    
    with col2:
        st.markdown(f"""
        * [INTEGRAL operations report @ ISDC](https://www.isdc.unige.ch/integral/operations/displayReport.cgi?rev={integral_time['SCWID'][:4]}) 
        * [INTEGRAL data consolidation report @ ISDC](https://www.isdc.unige.ch/integral/operations/displayConsReport.cgi?rev={integral_time['SCWID'][:4]})
        """
        )
    with col3:
        if source_coord is not None:
            st.markdown(f"""    
            With respect to source RA={source_coord.ra.deg:.3f},  DEC={source_coord.dec.deg:.3f}

            off-axis angle:{integral_sc['theta']:.3f}, phi={integral_sc['phi']:.3f}
            """)
except:
    st.markdown(f"""unable to deduce integral pointings (this is probably ok, some observations are out of pointings)""")

st.markdown(f"""
***
""")



@st.cache(ttl=360000, max_entries=100, persist=True)
def load_integral_observations(t0, scope_d, ra, dec):
    T = Time(t0, format='isot')
    t1 = Time(T.mjd - scope_d, format="mjd").isot[:10]
    t2 = Time(T.mjd + scope_d, format="mjd").isot[:10]


    scwlist = requests.get(f"https://www.astro.unige.ch/cdci/astrooda/dispatch-data/gw/timesystem/api/v1.0/scwlist/any/{t1}/{t2}?&ra={ra}&dec={dec}&radius=1005&min_good_isgri=1000&return_columns=SWID,RA_SCX,DEC_SCX,TSTART").json()
    return scwlist

if source_coord is not None:
    integral_observations = load_integral_observations(t0, scope_d, source_coord.ra.deg, source_coord.dec.deg)

import ivis

@st.cache(ttl=36000, max_entries=100, persist=True)
def load_ivis(ra, dec):
    return ivis.compute(target_ra=ra, target_dec=dec)

if source_coord is not None:
    with _lock:
        from matplotlib import pylab as plt
        import numpy as np
        from astropy.coordinates import SkyCoord

        fig = plt.figure(figsize=(15,3))

        C = SkyCoord(integral_observations['RA_SCX'], integral_observations['DEC_SCX'], unit='deg').separation(source_coord)
        plt.xlabel("")

        plt.scatter((np.array(integral_observations['TSTART']) - float(integral_time['IJD'])), C.deg)
        plt.axhspan(9, 15, alpha=0.1, color='y')
        plt.axhspan(0, 9, alpha=0.1, color='g')

        plt.title(f"off-axis angle for {source_coord} - might not be the GRB at T$_0$!")
        plt.xlabel(f"days since {t0}")
        st.pyplot(fig, clear_figure=True)

    try:
        visibility_map, esac_visibility_map = load_ivis(source_coord.ra.deg, source_coord.dec.deg)
    except Exception as e:
        print("\033[31mexception for accessing visibility origins:\033[0m", e)
        raise

    st.markdown("***")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("simplified high-resolution visbility")
        with _lock:
            from matplotlib import pylab as plt
            import numpy as np
            from astropy.coordinates import SkyCoord
            import integralvisibility

            target_mp = ivis.get_target_mp(source_coord.ra.deg, source_coord.dec.deg, 1, visibility_map)

            fig = plt.figure(figsize=(15,5))
            # integralvisibility.healpy.mollview(visibility_map,title="INTEGRAL visibility at NOW",cbar=False, fig=fig)

            integralvisibility.healtics.plot_with_ticks((visibility_map/visibility_map.max())*100, 
                                                    overplot=[[(target_mp,'r',target_mp.max()/10.)]],
                                                    vmin=0,
                                                    cmap="summer",
                                                    unit="%",
                                                    title="INTEGRAL visibility at NOW",
                                                    fig=fig)
            st.pyplot(fig, clear_figure=True)

    with col2:
        st.markdown("""
        INTEGRAL Visibility fetched from [ESAC](https://www.esa.int/About_Us/ESAC/Contact_ESAC) <a class="logo navbar-btn pull-left"
            target="_blank"
            href="https://www.cosmos.esa.int/web/integral/schedule-information"
            title="Home"> <img
            height="20px"
            src="https://www.cosmos.esa.int/o/CosmosTheme-theme/images/favicon.ico" alt="Home" />
        </a>""", unsafe_allow_html=True)
        with _lock:
            from matplotlib import pylab as plt
            import numpy as np
            from astropy.coordinates import SkyCoord
            import integralvisibility

            target_mp = ivis.get_target_mp(source_coord.ra.deg, source_coord.dec.deg, 1, visibility_map)

            fig = plt.figure(figsize=(15,5))
            # integralvisibility.healpy.mollview(visibility_map,title="INTEGRAL visibility at NOW",cbar=False, fig=fig)

            integralvisibility.healtics.plot_with_ticks((esac_visibility_map/esac_visibility_map.max())*100, 
                                                    overplot=[[(target_mp,'r',target_mp.max()/10.)]],
                                                    vmin=0,
                                                    cmap="summer",
                                                    unit="%",
                                                    title="INTEGRAL visibility at NOW",
                                                    fig=fig)
            st.pyplot(fig, clear_figure=True)



try:
    ibis_veto_lc = load_ibis_veto(t0, dt_s_download)
except Exception as e:
    ibis_veto_lc = None

try:
    isgri_events = deepcopy(load_isgri(t0, dt_s_download).copy())
except:
    isgri_events = None

try:
    spi_events = deepcopy(load_spi(t0, dt_s_download).copy())
except:
    spi_events = None

#isgri_image = load_isgri_image(t0).copy()


# st.sidebar.markdown('#### Whitened and band-passed data')
# whiten = st.sidebar.checkbox('Whiten?', value=True)
# freqrange = st.sidebar.slider('Band-pass frequency range (Hz)', min_value=10, max_value=2000, value=(30,400))


# -- Create sidebar for Q-transform controls
# st.sidebar.markdown('#### Q-tranform plot')
# vmax = st.sidebar.slider('Colorbar Max Energy', 10, 500, 25)  # min, max, default
# qcenter = st.sidebar.slider('Q-value', 5, 120, 5)  # min, max, default
# qrange = (int(qcenter*0.8), int(qcenter*1.2))

try:
    polar_lc = load_polar_lc(t0, dt_s)
except Exception as e:
    polar_lc = None

    

#-- Create a text element and let the reader know the data is loading.
strain_load_state = st.text('Loading data...this may take a minute')
try:
    lc = load_spiacs_lc(t0, dt_s_download)
except Exception as e:
    raise
    st.text('Data load failed.  Try a different time and detector pair.')
    raise st.script_runner.StopException



strain_load_state.text('Loading data...done!')

#-- Make a time series plot

# cropstart = t0-0.2
# cropend   = t0+0.1

# cropstart = t0 - dt
# cropend   = t0 + dt

st.subheader('Raw data')
#center = int(t0)
#lc = deepcopy(lc)


if False:
    with _lock:
        from matplotlib import pylab as plt
        import numpy as np

        levels=np.linspace(1,10, 100)
        d = isgri_image

        d[d<levels[0]] = levels[0]
        d[d>levels[-1]] = levels[-1]

        fig3 = plt.figure(figsize=(10,10))
        plt.contourf(d, levels=levels, cmap="jet")
        st.pyplot(fig3, clear_figure=True)


def rebin(S, n, offs=0, mean=True):
    N = int(len(S)/n)
    S = S[:N*n]

    if mean:
        return S.reshape(N, n).mean(1)
    else:
        return S.reshape(N, n).sum(1)


col1, col2 = st.columns(2)

with col1:
    with _lock:
        if lc is None:
            st.markdown("No SPI-ACS data could be retrieved! In the future, we will detect here if it's normal.")
        else:
            # fig1 = lc.crop(cropstart, cropend).plot()
            fig1 = plt.figure(figsize=(12,4))

            x = plt.errorbar(lc['TIME'], lc['RATE'], lc['ERROR'], ls="", alpha=0.8)
            plt.step(lc['TIME'], lc['RATE'], where='mid', c=x[0].get_color(), alpha=0.8)

            for rebin_n in [int(dt_rebin/0.05), ]:
                t = rebin(lc['TIME'], rebin_n)
                r = rebin(lc['RATE'], rebin_n)
                re = rebin(lc['ERROR']**2 , rebin_n)**0.5

                x = plt.errorbar(t, r, re, ls="", lw=3)
                plt.step(t, r, where='mid', c=x[0].get_color(), lw=3)
        

            #fig1 = cropped.plot()

            plt.title("INTEGRAL/SPI-ACS")
            plt.ylabel("counts/s")
            plt.xlabel(f"seconds since {t0}")
            plt.xlim([-dt_s, dt_s])

            st.pyplot(fig1, clear_figure=True)

    with st.expander("See notes"):
        st.markdown(f"""
    Total SPI-ACS rate, 100ms bins. Sensitive to whole sky, but less sensitive to directions around spacecraft pointing direction and especially the direction opposite to it. See above for the direction.
    """)

with col2:
    if polar_lc is not None:
        with _lock:
            polar_lc = np.array(polar_lc)
            #print("polar_lc", polar_lc)
            print("polar_lc", polar_lc.shape)
            print("polar_lc", np.array(polar_lc[0]).dtype)

            polar_lc = np.stack(polar_lc)

            # fig1 = lc.crop(cropstart, cropend).plot()
            fig2 = plt.figure(figsize=(12,4))

            t = Time(polar_lc['time'], format="unix").unix - Time(t0, format="isot").unix

            plt.title("POLAR")

            x = plt.errorbar( t, polar_lc['rate'], polar_lc['rate_err'], ls="")
            plt.ylabel("counts/s")
            plt.step( t, polar_lc['rate'], where='mid', c=x[0].get_color())
            #fig1 = cropped.plot()

            #fig1 = cropped.plot()

            plt.xlabel(f"seconds since {t0}")
            plt.xlim([-dt_s, dt_s])

            st.pyplot(fig2, clear_figure=True)
    else:
        st.markdown("POLAR data could not be retrieved! Is it out of the mission span?")

import gzip


@st.cache(ttl=360000, max_entries=100, persist=True)
def download_gbm_detector(t0, det):
    print("downloading gbm")
    for v in "00", "01", "02":
        try:
            url = f"https://heasarc.gsfc.nasa.gov/FTP/fermi/data/gbm/daily/{t0[:4]}/{t0[5:7]}/{t0[8:10]}/current/glg_ctime_{det}_{t0[2:4]}{t0[5:7]}{t0[8:10]}_v{v}.pha"
            print("url", url)
            c = requests.get(url).content
            f = fits.open(io.BytesIO(c))
            print("counts", f[2].data['COUNTS'].sum(0))
            print("time", f[2].data['TIME'])
            print("managed!")
            return np.copy(f[2].data['COUNTS'].sum(1)), np.copy(f[2].data['TIME'])
        except Exception as e:
            print("failed:", e)

    

@st.cache(ttl=3600, max_entries=10, persist=True)
def load_gbm(t0):
    d = {}
    print("loading gbm")
    for det in [f'n{i:x}' for i in range(12)] + [f'b{i:x}' for i in range(2)]:
         print(f"det {det} {t0}")
         d[f'rate_{det}']=download_gbm_detector(t0, det)[0]
         d[f'time_{det}']=download_gbm_detector(t0, det)[1]

    return d

    
if use_gbm:
    try:
        gbm = load_gbm(t0)
    except Exception as e:
        gbm = None


with col2:
    if ibis_veto_lc is not None:
        with _lock:
            polar_lc = np.array(polar_lc)
            
            # fig1 = lc.crop(cropstart, cropend).plot()
            fig3 = plt.figure(figsize=(12,4))

            t = (ibis_veto_lc['ijd'] - Time(t0).mjd + 51544)*24*3600

            plt.title("IBIS/Veto")

            x = plt.errorbar( t, ibis_veto_lc['rate'], ibis_veto_lc['rate']**0.5/8, ls="")

            plt.ylabel("counts/s")
            plt.step( t, ibis_veto_lc['rate'], where='mid', c=x[0].get_color())
            
            plt.xlabel(f"seconds since {t0}")
            plt.xlim([-dt_s, dt_s])

            st.pyplot(fig3, clear_figure=True)

        with st.expander("See notes"):
            st.markdown(f"""
        Total IBIS Veto rate, 8s bins. Sensitive primarily to directions opposite to the spacecraft pointing direction. See above for the direction.
        Note that this rate also contains periodic high bins, encoding different kind  of data. They should not be mistook for GRBs.
        """)
    else:
        st.markdown("IBIS Veto data could not be retrieved!")
        with st.expander("See notes"):
            st.markdown("Please consult the operations reports above\n")



#st.subheader('ISGRI')
#center = int(t0)
#lc = deepcopy(lc)

with col1:
    if isgri_events is not None:
        with _lock:
            # fig1 = lc.crop(cropstart, cropend).plot()
            fig2 = plt.figure(figsize=(12,4))

            t_ijd = isgri_events['TIME'][(isgri_events['ISGRI_ENERGY']>isgri_e1) & (isgri_events['ISGRI_ENERGY']<isgri_e2)]

            h = np.histogram((t_ijd - Time(t0, format="isot").mjd + 51544) * 24 * 3600, np.linspace(-dt_s, dt_s, 300))

            plt.step(
                (h[1][1:] + h[1][:-1]), 
                h[0]/(h[1][1:] - h[1][:-1]))
            #fig1 = cropped.plot()

            plt.xlabel(f"seconds since {t0}")
            plt.ylabel("counts / s (full energy range)")
            plt.title("ISGRI total rate")
            plt.xlim([-dt_s, dt_s])

            st.pyplot(fig2, clear_figure=True)

        with st.expander("See notes"):
            st.markdown(f"""
        Total ISGRI, 300 bins in the requested interval. Sensitive primarily to directions within 80 deg from spacecraft pointing direction. See above for the direction.
        """)
    else:
        st.markdown("ISGRI data could not be retrieved!")
        with st.expander("See notes"):
            st.markdown("Please consult the operations reports above\n" )


with col2:
    if spi_events is not None:
        with _lock:
            # fig1 = lc.crop(cropstart, cropend).plot()
            fig2 = plt.figure(figsize=(12,4))

            t_ijd = spi_events['TIME'][(spi_events['ENERGY']>isgri_e1) & (spi_events['ENERGY']<isgri_e2)]

            h = np.histogram((t_ijd - Time(t0, format="isot").mjd + 51544) * 24 * 3600, np.linspace(-dt_s, dt_s, 300))

            plt.step(
                (h[1][1:] + h[1][:-1]), 
                h[0]/(h[1][1:] - h[1][:-1]))
            #fig1 = cropped.plot()

            plt.xlabel(f"seconds since {t0}")
            plt.ylabel("counts / s (full energy range)")
            plt.title("SPI total rate")
            plt.xlim([-dt_s, dt_s])

            st.pyplot(fig2, clear_figure=True)

        with st.expander("See notes"):
            st.markdown(f"""
        Total SPI, 300 bins in the requested interval. Sensitive primarily to directions within 80 deg from spacecraft pointing direction. See above for the direction.
        """)
    else:
        st.markdown("SPI data could not be retrieved!")
        with st.expander("See notes"):
            st.markdown("Please consult the operations reports above\n" )



if use_gbm and gbm is not None:
    with _lock:
        # fig1 = lc.crop(cropstart, cropend).plot()
        fig2 = plt.figure(figsize=(12,6))
        #h = np.histogram((isgri_events['TIME'] - Time(t0, format="isot").mjd + 51544) * 24 * 3600, np.linspace(-dt_s, dt_s, 300))

        for det in [k.replace('time_', '') for k in gbm.keys() if k.startswith('time_')]:
            t = gbm['time_'+det] - float(t0_xtime['time_in_sf'])
            
            # - Time(t0, format='isot')).seconds

            m = t>-dt_s
            m &= t<dt_s

            plt.step(t[m], gbm['rate_'+det][m], label=det)
            #fig1 = cropped.plot()

        plt.xlabel(f"seconds since {t0}")
        plt.ylabel("counts / s (full energy range)")
        plt.title("GBM")

        plt.legend()
        #plt.xlim([-dt_s, dt_s])

        st.pyplot(fig2, clear_figure=True)

    with st.expander("See notes"):
        st.markdown(f"""
    Total ISGRI, 300 bins in the requested interval. Sensitive primarily to directions within 80 deg from spacecraft pointing direction. See above for the direction.
    """)
else:
    st.markdown("GBM data could not be retrieved!")
    # with st.beta_expander("See notes"):
    #     st.markdown("Please consult the operations reports:\n" + integral_reports)



# -- Try whitened and band-passed plot
# -- Whiten and bandpass data
# st.subheader('Whitened and Band-passed Data')

# if whiten:
#     white_data = strain.whiten()
#     bp_data = white_data.bandpass(freqrange[0], freqrange[1])
# else:
#     bp_data = strain.bandpass(freqrange[0], freqrange[1])

# bp_cropped = bp_data.crop(cropstart, cropend)

# with _lock:
#     fig3 = bp_cropped.plot()
#     st.pyplot(fig3, clear_figure=True)

# # -- Allow data download
# download = {'Time':bp_cropped.times, 'Strain':bp_cropped.value}
# df = pd.DataFrame(download)
# csv = df.to_csv(index=False)
# b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
# href = f'<a href="data:file/csv;base64,{b64}">Download Data as CSV File</a>'
# st.markdown(href, unsafe_allow_html=True)

# # -- Notes on whitening
# with st.beta_expander("See notes"):
#     st.markdown("""
#  * Whitening is a process that re-weights a signal, so that all frequency bins have a nearly equal amount of noise. 
#  * A band-pass filter uses both a low frequency cutoff and a high frequency cutoff, and only passes signals in the frequency band between these values.

# See also:
#  * [Signal Processing Tutorial](https://share.streamlit.io/jkanner/streamlit-audio/main/app.py)
# """)


# st.subheader('Q-transform')

# hq = strain.q_transform(outseg=(t0-dt, t0+dt), qrange=qrange)

# with _lock:
#     fig4 = hq.plot()
#     ax = fig4.gca()
#     fig4.colorbar(label="Normalised energy", vmax=vmax, vmin=0)
#     ax.grid(False)
#     ax.set_yscale('log')
#     ax.set_ylim(bottom=15)
#     st.pyplot(fig4, clear_figure=True)


st.subheader("About this app")
st.markdown("""
This app displays data from INTEGRAL and POLAR, downloaded from https://www.astro.unige.ch/mmoda/ .

""")


if auth is not None:
    # TODO!
    event_reporting_gcn_id = 9999

    @st.cache(ttl=5, max_entries=10, persist=False)   #-- Magic command to cache data
    def load_igcn(t0, ra, dec, name):
        url = (f"https://oda-workflows-gcn-circular-integral-ul.odahub.io/api/v1.0/get/gcn?"
               f"datasource=nrt&gcn_number={event_reporting_gcn_id}&name={name}&t0_utc={t0}&ra={ra:.5g}&dec={dec:.5g}&radius=5&healpix_url=&event_kind=UNKNOWN&test=0&_async_request=yes"
               )
        #st.markdown(url)
        r = requests.get(url, auth=auth)    

        try:
            return r.json(), url
        except:
            return r.text, url


    if source_coord is not None:
        if st.sidebar.checkbox('Load INTEGRAL GCN'):
            igcn, url = load_igcn(t0, source_coord.ra.deg, source_coord.dec.deg, source_name)

            json.dump(igcn, open("igcn.json", "w"))

            try:
                st.markdown(f"""
                    | | |
                    | :--: | :--: | 
                    | GCN workflow status | [{igcn['workflow_status']}]({url}) |
                    """)
            except:
                st.markdown(f"{igcn[0]} {igcn[1]}")


            if igcn['workflow_status'] == "done":
                st.markdown(igcn['data']['output'].keys())

                import base64

                st.markdown(igcn['data']['output']['gcn_html'], unsafe_allow_html=True)

                st.image(base64.b64decode(igcn['data']['output']['sens_map_soft_png_content']))

                # st.image(base64.b64decode(igcn['data']['output']['sens_map_hard_png']))

