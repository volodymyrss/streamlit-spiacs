import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import requests, os

from copy import deepcopy
import base64

from astropy.time import Time

# Use the non-interactive Agg backend, which is recommended as a
# thread-safe backend.
# See https://matplotlib.org/3.3.2/faq/howto_faq.html#working-with-threads.
import matplotlib as mpl
mpl.use("agg")

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

st.set_page_config(page_title=apptitle, page_icon=":eyeglasses:")

# -- Default detector list
#detectorlist = ['H1','L1', 'V1']

# Title the app
st.title('SPI-ACS, POLAR + ISGRI Quickview')

st.markdown("""
 * Use the menu at left to select data and set plot parameters
 * Your plots will appear below
""")

import oda_api.api

from astropy.time import Time

@st.cache(ttl=3600, max_entries=10, persist=True)   #-- Magic command to cache data
def load_lc(t0, dt_s):
    _t0 = Time(t0, format="isot")

    disp = oda_api.api.DispatcherAPI(url="https://www.astro.unige.ch/mmoda/dispatch-data/")

    t1_isot = Time(_t0.mjd - dt_s/24/3600, format="mjd").isot
    t2_isot = Time(_t0.mjd + dt_s/24/3600, format="mjd").isot

    print("t1, t2", t1_isot, t2_isot)

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

    print("lc", lc)

    return lc


@st.cache(ttl=3600, max_entries=10, persist=True)   #-- Magic command to cache data
def load_polar_lc(t0, dt_s):
    _t0 = Time(t0, format="isot")

    disp = oda_api.api.DispatcherAPI(url="https://www.astro.unige.ch/mmoda/dispatch-data/")

    t1_isot = Time(_t0.mjd - dt_s/24/3600, format="mjd").isot
    t2_isot = Time(_t0.mjd + dt_s/24/3600, format="mjd").isot

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


import integralclient as ic

@st.cache(ttl=3600, max_entries=10, persist=True)   #-- Magic command to cache data
def load_integral_time(t0):
    _t0 = Time(t0, format="isot")
    d = ic.converttime("UTC", _t0.isot, "ANY")
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

    import io
    from astropy.io import fits

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

@st.cache(ttl=3600, max_entries=10, persist=True)   #-- Magic command to cache data
def load_ibis_veto(t0, dt_s):
    t = requests.get(f"https://www.astro.unige.ch/cdci/astrooda/dispatch-data/gw/integralhk/api/v1.0/genlc/IBIS_VETO/{t0}/{dt_s}").json()
    print("\033[31m", t, "\033[0m")
    d = np.genfromtxt(io.StringIO(t), names=['ijd', 'time', 'rate', 'x'])
    return d

st.sidebar.markdown("## Select Data Time")

# -- Get list of events
# find_datasets(catalog='GWTC-1-confident',type='events')
# eventlist = datasets.find_datasets(type='events')
# eventlist = [name.split('-')[0] for name in eventlist if name[0:2] == 'GW']
# eventset = set([name for name in eventlist])
# eventlist = list(eventset)
# eventlist.sort()

#-- Set time by GPS or event
select_event = st.sidebar.selectbox('How do you want to find data?',
                                    ['By UTC', 'By event name'])

if select_event == 'By UTC':
    # -- Set a GPS time:        
    t0 = st.sidebar.text_input('UTC', '2017-01-05T06:14:07').strip()    # -- GW150914
    #t0 = float(str_t0)

    # st.sidebar.markdown("""
    # Example times in the H1 detector:
    # * 1126259462.4    (GW150914) 
    # * 1187008882.4    (GW170817) 
    # * 933200215       (hardware injection)
    # * 1132401286.33   (Koi Fish Glitch) 
    # """)

else:
    #t0 = st.sidebar.text_input('UTC', '2008-03-19T06:12:44')    # -- GW150914


    eventlist = {
        "GRB170817A": '2017-08-17T12:41:00',
        "GRB080319B": '2008-03-19T06:12:44',
        "GRB120711A": "2012-07-11T02:45:30.0",
    }
    
    chosen_event = st.sidebar.selectbox('Select Event', sorted(list(eventlist.keys())))
    
    t0 = eventlist[chosen_event]
    
    st.subheader(chosen_event)
    st.write('GPS:', t0)
    

    
#-- Choose detector as H1, L1, or V1
#detector = st.sidebar.selectbox('Detector', detectorlist)

# -- Create sidebar for plot controls
st.sidebar.markdown('## Set Plot Parameters')
dtboth = st.sidebar.slider('Time Range (seconds)', 0.5, 1000.0, 50.0)  # min, max, default
dt_s = dtboth / 2.0



integral_time = load_integral_time(t0)

st.markdown(f"""
INTEGRAL ScW: {integral_time['SCWID']}

* [INTEGRAL operations report @ ISDC](https://www.isdc.unige.ch/integral/operations/displayReport.cgi?rev={integral_time['SCWID'][:4]}) 
* [INTEGRAL data consolidation report @ ISDC](https://www.isdc.unige.ch/integral/operations/displayConsReport.cgi?rev={integral_time['SCWID'][:4]})
""")

try:
    ibis_veto_lc = load_ibis_veto(t0, dt_s)
except Exception as e:
    ibis_veto_lc = None

#isgri_events = deepcopy(load_isgri(t0, dt_s).copy())
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
    raise
    polar_lc = None

    

#-- Create a text element and let the reader know the data is loading.
strain_load_state = st.text('Loading data...this may take a minute')
try:
    lc = load_lc(t0, dt_s)
except Exception as e:
    raise
    st.text('Data load failed.  Try a different time and detector pair.')
    st.text('Problems can be reported to gwosc@igwn.org')
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


def rebin(S, n):
    S = S[:]

with _lock:
    # fig1 = lc.crop(cropstart, cropend).plot()
    fig1 = plt.figure(figsize=(12,6))

    x = plt.errorbar(lc['TIME'], lc['RATE'], lc['ERROR'] * 20**0.5, ls="")
    plt.step(lc['TIME'], lc['RATE'], where='mid', c=x[0].get_color())
    #fig1 = cropped.plot()

    plt.title("INTEGRAL/SPI-ACS")
    plt.ylabel("counts/s")
    plt.xlabel(f"seconds since {t0}")

    st.pyplot(fig1, clear_figure=True)

if polar_lc is not None:
    with _lock:
        polar_lc = np.array(polar_lc)
        #print("polar_lc", polar_lc)
        print("polar_lc", polar_lc.shape)
        print("polar_lc", np.array(polar_lc[0]).dtype)

        polar_lc = np.stack(polar_lc)

        # fig1 = lc.crop(cropstart, cropend).plot()
        fig2 = plt.figure(figsize=(12,6))

        t = Time(polar_lc['time'], format="unix").unix - Time(t0, format="isot").unix

        plt.title("POLAR")

        x = plt.errorbar( t, polar_lc['rate'], polar_lc['rate_err'], ls="")
        plt.ylabel("counts/s")
        plt.step( t, polar_lc['rate'], where='mid', c=x[0].get_color())
        #fig1 = cropped.plot()

        #fig1 = cropped.plot()

        plt.xlabel(f"seconds since {t0}")

        st.pyplot(fig2, clear_figure=True)



if ibis_veto_lc is not None:
    with _lock:
        polar_lc = np.array(polar_lc)
        
        # fig1 = lc.crop(cropstart, cropend).plot()
        fig3 = plt.figure(figsize=(12,6))

        t = Time(ibis_veto_lc['time'], format="unix").unix - Time(t0, format="isot").unix

        plt.title("IBIS/Veto")

        x = plt.errorbar( t, ibis_veto_lc['rate'], ibis_veto_lc['rate']**0.5/8, ls="")

        plt.ylabel("counts/s")
        plt.step( t, ibis_veto_lc['rate'], where='mid', c=x[0].get_color())
        
        plt.xlabel(f"seconds since {t0}")

        st.pyplot(fig3, clear_figure=True)


st.subheader('ISGRI')
#center = int(t0)
#lc = deepcopy(lc)


if False:
    with _lock:
        # fig1 = lc.crop(cropstart, cropend).plot()
        fig2 = plt.figure()
        h = np.histogram((isgri_events['TIME'] - Time(t0, format="isot").mjd + 51544) * 24 * 3600, np.linspace(-dt_s, dt_s, 100))

        plt.step(h[1][:-1], h[0])
        #fig1 = cropped.plot()
        st.pyplot(fig2, clear_figure=True)


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


with st.beta_expander("See notes"):

    st.markdown("""
A Q-transform plot shows how a signal’s frequency changes with time.

 * The x-axis shows time
 * The y-axis shows frequency

The color scale shows the amount of “energy” or “signal power” in each time-frequency pixel.

A parameter called “Q” refers to the quality factor.  A higher quality factor corresponds to a larger number of cycles in each time-frequency pixel.  

For gravitational-wave signals, binary black holes are most clear with lower Q values (Q = 5-20), where binary neutron star mergers work better with higher Q values (Q = 80 - 120).

See also:

 * [GWpy q-transform](https://gwpy.github.io/docs/stable/examples/timeseries/qscan.html)
 * [Reading Time-frequency plots](https://labcit.ligo.caltech.edu/~jkanner/aapt/web/math.html#tfplot)
 * [Shourov Chatterji PhD Thesis](https://dspace.mit.edu/handle/1721.1/34388)
""")


st.subheader("About this app")
st.markdown("""
This app displays data from INTEGRAL and POLAR, downloaded from https://www.astro.unige.ch/mmoda/ .

""")

