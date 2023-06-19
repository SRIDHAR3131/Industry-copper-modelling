from streamlit_lottie import st_lottie
import requests
#"https://assets1.lottiefiles.com/private_files/lf30_kit9njnq.json"
def lottie_home0(url_link0):
    url = requests.get(url_link0)
    url_json = dict()
    if url.status_code == 200:
        url_json = url.json()
    else:
        print("Error in URL")
    st_lottie(url_json,
              reverse=False,
              height=True,
              width=True,
              speed=1.5,
              loop=True,
              quality='high'
              )
    return  st_lottie


def lottie_home1(url_link1):
    url = requests.get(url_link1)
    url_json = dict()
    if url.status_code == 200:
        url_json = url.json()
    else:
        print("Error in URL")
    st_lottie(url_json,
              reverse=False,
              height=True,
              width=True,
              speed=2.5,
              loop=True,
              quality='high'
              )
    return  st_lottie



def lottie_price1(url_link2):
    url = requests.get(url_link2)
    url_json = dict()
    if url.status_code == 200:
        url_json = url.json()
    else:
        print("Error in URL")
    st_lottie(url_json,
              reverse=False,
              height=True,
              width=True,
              speed=8,
              loop=True,
              quality='high'
              )
    return  st_lottie


def lottie_status1(url_link3):
    url = requests.get(url_link3)
    url_json = dict()
    if url.status_code == 200:
        url_json = url.json()
    else:
        print("Error in URL")
    st_lottie(url_json,
              reverse=False,
              height=200,
              width=240,
              speed=1,
              loop=True,
              quality='high'
              )
    return  st_lottie

def lottie_status2(url_link4):
    url = requests.get(url_link4)
    url_json = dict()
    if url.status_code == 200:
        url_json = url.json()
    else:
        print("Error in URL")
    st_lottie(url_json,
              reverse=False,
              height=200,
              width=100,
              speed=1.5,
              loop=True,
              quality='high'
              )
    return  st_lottie