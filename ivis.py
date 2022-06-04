import requests
import integralvisibility
from matplotlib import pylab
from astropy.coordinates import SkyCoord
import integralclient as ic
import datetime
import numpy as np

class ServiceResponseProblem(Exception):
    pass

def get_target_mp(target_ra, target_dec, target_radius, basemp):
    radec=integralvisibility.healpy.pix2ang(integralvisibility.healpy.npix2nside(len(basemp)), range(len(basemp)), lonlat=True)
    ra,dec=radec

    dist_ra=(ra - target_ra)
    dist_ra[abs(dist_ra)>180]-=360

    target_mp=np.exp(
        -0.5*(dist_ra/target_radius)**2  \
        -0.5*((dec - target_dec)/target_radius )**2
    )

    target_mp/=np.sum(target_mp)

    return target_mp


def compute(
    tstart_utc="unknown",
    duration_days=10.01,
    target_loc="point",
    target_ra=83.01,
    target_dec=22.02,
    target_radius=1.,
    enable_esac=1
    ):

    if tstart_utc == "unknown":
        tstart_utc = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")


    tstart_ijd = float(ic.converttime("UTC",tstart_utc,"IJD"))
    tstop_ijd = tstart_ijd+duration_days

    vis=integralvisibility.Visibility(minsolarangle=40)
    vis.minsolarangle=40

    mp = vis.for_time(tstart_ijd,nsides=128)

    sc = ic.get_sc(tstart_utc)
    sc

    try:
        now_point = SkyCoord(sc['scx']['ra'], sc['scx']['dec'], unit='deg')
    except Exception as e:        
        raise ServiceResponseProblem(f"problem with sc response for {tstart_utc}: {sc}")

    # #http://integral.esac.esa.int/isocweb/tvp.html?revolution=2050&action=perRevolution
    # r=requests.get("http://integral.esac.esa.int/isocweb/tvp.html",
    #             params=dict(
    #                     startDate=tstart_utc[8:10]+"-"+tstart_utc[5:7]+"-"+tstart_utc[:4],                    
    #                     duration="12.600",
    #                     dither="R",
    #                     action="predict",
    #                     endDate="31-12-2019",
    #                     coordinates="equatorial",
    #                     ra=target_ra,
    #                     dec=target_dec,
    #                     format="json"))

    # try:
    #     if not enable_esac: raise RuntimeError
        
    #     visibility = r.json()
        
    #     #visibility
    #     pylab.figure()

    #     for interval in visibility['INTERVALS'][:10]:
    #         print(interval)
    #         i1=float(ic.converttime("UTC",interval['start'][:-4].replace(" ","T"),"IJD"))
    #         i2=float(ic.converttime("UTC",interval['end'][:-4].replace(" ","T"),"IJD"))

    #         print(i1,i2)

    #         pylab.axvspan(i1,i2,alpha=0.5)

    #     pylab.axvspan(tstart_ijd,tstop_ijd,alpha=0.5,color='g')
    #     pylab.xlabel("IJD")

    #     pylab.savefig('visibility_intervals.png')

    # except:
    #     print(r.text)


    esac_visibility_rev = ic.converttime('UTC',tstart_utc,'REVNUM')

    print("will fetch ESOC visibility map for rev",esac_visibility_rev)

    try:
        r=requests.get("http://integral.esac.esa.int/isocweb/tvp.html",
                params=dict(
                        revolution=esac_visibility_rev,
                        action='perRevolution',
                        ))

        esac_visibility_map = r.json()
        print("ESAC VM success")
    except:
        esac_visibility_map = None
        print("ESAC visibility failed:", r.text)


    import healpy
    import numpy as np

    mp_ra, mp_dec = integralvisibility.healpy.pix2ang(128,range(integralvisibility.healpy.nside2npix(128)),lonlat=True)

    esac_mp = np.zeros_like(mp_ra)

    try:
        if not enable_esac: raise RuntimeError
        for sky_bin in esac_visibility_map['sky_bins'][:10000]:
            #print(sky_bin)

            m = (mp_ra > float(sky_bin['top_left_corner'][0])) & (mp_ra < float(sky_bin['top_right_corner'][0]))
            m &= (mp_dec < float(sky_bin['top_left_corner'][1])) & (mp_dec > float(sky_bin['bottom_left_corner'][1]))

            esac_mp[m]=float(sky_bin['duration'])
    except Exception as e:
        print("ESAC visibility failed:", e)
        esac_mp = mp


    return mp, esac_mp

#     import healpy
#     import numpy as np

#     mp_ra, mp_dec = integralvisibility.healpy.pix2ang(128,range(integralvisibility.healpy.nside2npix(128)),lonlat=True)

#     esac_mp = np.zeros_like(mp_ra)

#     try:
#         if not enable_esac: raise RuntimeError
#         for sky_bin in esac_visibility_map['sky_bins'][:10000]:
#             #print(sky_bin)

#             m = (mp_ra > float(sky_bin['top_left_corner'][0])) & (mp_ra < float(sky_bin['top_right_corner'][0]))
#             m &= (mp_dec < float(sky_bin['top_left_corner'][1])) & (mp_dec > float(sky_bin['bottom_left_corner'][1]))

#             esac_mp[m]=float(sky_bin['duration'])
#     except Exception as e:
#         print("ESAC visibility failed:", e)
#         esac_mp = mp

#     def total_slew_time_s(distance_deg):
#     total_slew_options = []
    
#     slew_duration_s = distance_deg*60*60/185.
    
#     total_slew_options.append(slew_duration_s)
    
#     if distance_deg>45:
#         rwb_duration_s = 22*60. 
#         total_slew_options.append(slew_duration_s + rwb_duration_s)
    
# #    return total_slew_options

#     def total_reaction_time_s(distance_deg):
#         return [ (x + 3600*2) for x in total_slew_time_s(distance_deg)]
        
#     total_reaction_time_s(10),total_reaction_time_s(50)

#     visibility_summary = {}

#     for kind, vismp in [("std", mp), ("esac", esac_mp)]:
#         visibility_summary[kind]={}
#         vis = visibility_summary[kind]
        
#         nsides = healpy.npix2nside(len(vismp))

#     #  integralvisibility.healpy.mollview(vismp,title="INTEGRAL visibility at "+tstart_utc[:19],cbar=False)
#     #  integralvisibility.healpy.graticule()
#     #  pylab.savefig("visibility_"+kind+".png")

#         radec=integralvisibility.healpy.pix2ang(integralvisibility.healpy.npix2nside(len(vismp)), range(len(vismp)), lonlat=True)
#         ra,dec=radec
        
#         sep_from_now_deg = np.transpose(SkyCoord(*radec,unit='deg').separation(now_point).deg)
        
#         dist_ra=(ra - target_ra)
#         dist_ra[abs(dist_ra)>180]-=360

#         if target_loc == "point":
            
#                 target_mp=np.exp(
#                     -0.5*(dist_ra/target_radius)**2  \
#                     -0.5*((dec - target_dec)/target_radius )**2
#                 )

#                 target_mp/=np.sum(target_mp)
            
        
            

#         integralvisibility.healtics.plot_with_ticks((vismp/vismp.max())*100, 
#                                                     overplot=[[(target_mp,'r',target_mp.max()/10.)]],
#                                                     vmin=0,
#                                                     cmap="summer",
#                                                     unit="%",
#                                                     title="INTEGRAL visibility at "+tstart_utc[:19])

#     #     integralvisibility.healpy.mollview(vismp/vismp.max() + target_mp/target_mp.max(),title="INTEGRAL visibility at "+tstart_utc[:19],cbar=False)
#     #     integralvisibility.healpy.graticule()

#         vis['on_peak'] = vismp[target_mp.argmax()]
#         vis['on_peak_frac'] = vis['on_peak'] / vismp.max()

#         vis['on_probability'] = target_mp[vismp > vismp.max()/2.].sum()

#         #vis['visible_peak_ra'] = healpy.pix2ang(nsides, target_mp[vismp > vismp.max()/2.].argmax(),)
#         vis['visible'] = vis['on_probability']
        
#         def argmax_masked(x,y):
#             mi = x[y].argmax()
#             print("mi",mi)
#             idx = np.array(range(len(x)))
#             print("idx",idx)
#             return idx[y][mi]

#         vis['peak_of_target'] = list(np.transpose(radec)[target_mp.argmax()])
#         vis['peak_of_visible'] = list(np.transpose(radec)[argmax_masked(target_mp, vismp > vismp.max()/2.)])


#         vis['points'] = [
#             dict(descr="total visble",prob=vis['on_probability']),
#             dict(descr="peak",ra=vis['peak_of_target'][0],dec=vis['peak_of_target'][1]),
#             dict(descr="visible peak",ra=vis['peak_of_visible'][0],dec=vis['peak_of_visible'][1]),
#         ]
        
        
#         colors=['r','m','k','b','g','c','y','w','grey']
#         for limrange in None, 45, 90:
            
#             limmod = "lim%.3lgdeg_"%limrange if limrange is not None else ""
#             limdescr = " (%.3lg deg from now)"%limrange if limrange is not None else ""
            
            
#             for scale, descr in [(5, "best for staring"), (15, "best for HEX"), (30, "best for 5x5")]:
#                 c=colors.pop()
#                 target_sm = healpy.smoothing(target_mp/target_mp.max(), sigma=scale/180.*np.pi,verbose=False)
#                 target_sm/=sum(target_sm)

#                 print(sum(target_sm),sum(target_mp))
                            
#                 mask_visible = (vismp > vismp.max()/2.)
#                 if limrange is not None:
#                     mask_visible &= (sep_from_now_deg < limrange)
                    
#                 vis[limmod+'best_visible_%ideg'%scale]=list(np.transpose(radec)[argmax_masked(target_sm, mask_visible)])
#                 vis[limmod+'best_visible_prob_%ideg'%scale]=max(target_sm[mask_visible])*scale*scale*4*np.pi*2  

#                 true_peak = SkyCoord(vis['peak_of_target'][0],vis['peak_of_target'][1],unit="deg")

#                 vis['points'] += [dict(
#                     alt_descr="peak at %i deg scale"%scale+limdescr,            
#                     descr = descr+limdescr,            
#                     ra=vis[limmod+'best_visible_%ideg'%scale][0],
#                     dec=vis[limmod+'best_visible_%ideg'%scale][1],
#                     prob=vis[limmod+'best_visible_prob_%ideg'%scale],
#                     distance_to_true_peak_deg=SkyCoord(vis[limmod+'best_visible_%ideg'%scale][0], vis[limmod+'best_visible_%ideg'%scale][1],unit="deg").separation(true_peak).deg,
#                     distance_to_now=SkyCoord(vis[limmod+'best_visible_%ideg'%scale][0], vis[limmod+'best_visible_%ideg'%scale][1],unit="deg").separation(now_point).deg
#                 )]


#                 def c_ra(x):
#                     if x<180:
#                         return -x
#                     return 360-x

#                 pylab.scatter(
#                     c_ra(vis[limmod+'best_visible_%ideg'%scale][0])/180.*np.pi,
#                     vis[limmod+'best_visible_%ideg'%scale][1]/180.*np.pi,
#                     s=100,c=c)

#                 pylab.text(
#                     c_ra((-5+vis[limmod+'best_visible_%ideg'%scale][0]))/180.*np.pi,
#                     (5+vis[limmod+'best_visible_%ideg'%scale][1])/180.*np.pi,
#                     "%i deg"%scale,
#                     size=10,color=c)
            
        
#         pylab.scatter(
#             c_ra(sc['scx']['ra'])/180.*np.pi,
#             sc['scx']['dec']/180.*np.pi,
#             marker='x',
#             s=200,c='k',
#             label='FoV')
        
#         #pylab.legend()

        

            
#         print(visibility_summary)


#         pylab.savefig("skymap_visibility_"+kind+".png")



#     template="""
# <html>
# <head>
# </head>
# <body>

# <table style="border-collapse: separate;border-spacing: 2px;">


# <tr style="background-color:#ADD8E6">
# <td align="center"></td>
# <td align="center">RA</td>
# <td align="center">Dec</td>
# <td align="center">Probability</td>
# <!--<td align="center">To Peak</td>-->
# <td align="center">From current</td>
# </tr>

# {% for point in points %}

# <tr>
# <td style="text-align:center" >
# <div data-toggle="{{ point.uid }}_loc_tooltip" title="{{ point.alt_descr }}" style="display:inline-block;margin-right:10px;">
# {{ point.descr }}
# </div>
# </td>

# {% if point.ra %}
# <td style="text-align:center" >{{ point.ra | round(1) }}</td>
# <td style="text-align:center" >{{ point.dec | round(1)}}</td>
# {% else %}
# <td style="text-align:center" ></td>
# <td style="text-align:center" ></td>
# {% endif %}

# {% if point.prob %}
# <td style="text-align:center" >{{ (100*point.prob) | round(1)}}%</td>
# {% else %}
# <td style="text-align:center" ></td>
# {% endif %}

# <!--
# {% if point.distance_to_true_peak_deg %}
# <td style="text-align:center" >{{ point.distance_to_true_peak_deg | round(1)}} deg</td>
# {% else %}
# <td style="text-align:center" ></td>
# {% endif %}
# -->


# {% if point.distance_to_now %}
# <td style="text-align:center" >{{ point.distance_to_now | round(1)}} deg</td>
# {% else %}
# <td style="text-align:center" ></td>
# {% endif %}


# </tr>

# {% endfor %}

# <tr style="background-color:#ADD8E6">
# <td colspan=5 height="5px"></td>
# </tr>


# </table>

# </body>
# </html>
#     """

#     from IPython.core.display import display, HTML
#     from jinja2 import Environment, BaseLoader

#     rtemplate = Environment(loader=BaseLoader).from_string(template)


#     data_html = rtemplate.render(points=visibility_summary['esac']['points'])

#     open("visibility_card.html","w").write(data_html)

#     display(HTML(data_html))