# PyTracking Model Zoo

Here, we provide a number of tracker models trained using PyTracking. We also report the results
of the models on standard tracking datasets.  

## Models

<table>
  <tr>
    <th>Model</th>
    <th>VOT18<br>EAO (%)</th>
    <th>OTB100<br>AUC (%)</th>
    <th>NFS<br>AUC (%)</th>
    <th>UAV123<br>AUC (%)</th>
    <th>LaSOT<br>AUC (%)</th>
    <th>TrackingNet<br>AUC (%)</th>
    <th>GOT-10k<br>AO (%)</th>
    <th>Links</th>
  </tr>
  <tr>
    <td>ATOM</td>
    <td>0.401</td>
    <td>66.3</td>
    <td>58.4</td>
    <td>64.2</td>
    <td>51.5</td>
    <td>70.3</td>
    <td>55.6</td>
    <td><a href="https://drive.google.com/open?id=1VNyr-Ds0khjM0zaq6lU-xfY74-iWxBvU">model</a></td>
  </tr>
  <tr>
    <td>DiMP-18</td>
    <td>0.402</td>
    <td>66.0</td>
    <td>61.0</td>
    <td>64.3</td>
    <td>53.5</td>
    <td>72.3</td>
    <td>57.9</td>
    <td><a href="https://drive.google.com/open?id=1MAjrRJDCbL0DSjUKFyDkUuYS1-cYBNjk">model</a></td>
  </tr>
  <tr>
    <td>DiMP-50</td>
    <td>0.440</td>
    <td>68.4</td>
    <td>61.9</td>
    <td>65.3</td>
    <td>56.9</td>
    <td>74.0</td>
    <td>61.1</td>
    <td><a href="https://drive.google.com/open?id=1qgachgqks2UGjKx-GdO1qylBDdB1f9KN">model</a></td>
  </tr>
  <tr>
    <td>PrDiMP-18</td>
    <td>0.385</td>
    <td>68.0</td>
    <td>63.3</td>
    <td>65.3</td>
    <td>56.4</td>
    <td>75.0</td>
    <td>61.2</td>
    <td><a href="https://drive.google.com/open?id=1ycm3Uu63j-uCkz4qt0SG6rY_k5UFlhVo">model</a></td>
  </tr>
  <tr>
    <td>PrDiMP-50</td>
    <td>0.442</td>
    <td>69.6</td>
    <td>63.5</td>
    <td>68.0</td>
    <td>59.8</td>
    <td>75.8</td>
    <td>63.4</td>
    <td><a href="https://drive.google.com/open?id=1zbQUVXKsGvBEOc-I1NuGU6yTMPth_aI5">model</a></td>
  </tr>
  <tr>
    <td>SuperDimp</td>
    <td>-</td>
    <td>70.1</td>
    <td>64.7</td>
    <td>68.1</td>
    <td>63.1</td>
    <td>78.1</td>
    <td>-</td>
    <td><a href="https://drive.google.com/open?id=1qDptswis2FxihLRYLVRGDvx6aUoAVVLv">model</a></td>
  </tr>
</table>

## Raw Results
The raw results can be downloaded automatically using the [download_results](pytracking/util_scripts/download_results.py) script.
You can also download and extract them manually from https://drive.google.com/open?id=1Sacgh5TZVjfpanmwCFvKkpnOA7UHZCY0. The folder ```benchmark_results``` contains raw results for all datasets except VOT. These results can be analyzed using the [analysis](pytracking/analysis) module in pytracking. Check [pytracking/notebooks/analyze_results.ipynb](pytracking/notebooks/analyze_results.ipynb) for examples on how to use the analysis module. The folder ```packed_results``` contains packed results for TrackingNet and GOT-10k, which can be directly evaluated on the official evaluation servers, as well as the VOT results. 

The raw results are in the format [top_left_x, top_left_y, width, height]. 
Due to the stochastic nature of the trackers, the results reported here are an average over multiple runs. 
For OTB-100, NFS, UAV123, and LaSOT, the results were averaged over 5 runs. For VOT2018, 15 runs were used 
as per the VOT protocol. As TrackingNet results are obtained using the online evaluation server, only a 
single run was used for TrackingNet. For GOT-10k, 3 runs are used as per protocol.

## Plots
The success plots for our trained models on the standard tracking datasets are shown below.  

#### LaSOT
![LaSOT](pytracking/.figs/LaSOT.png)  

#### OTB-100
![OTB-100](pytracking/.figs/OTB-100.png)  

#### NFS
![NFS](pytracking/.figs/NFS.png)  

#### UAV123
![UAV123](pytracking/.figs/UAV123.png)  
