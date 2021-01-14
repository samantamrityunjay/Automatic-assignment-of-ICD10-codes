<h1> Explainable AI: ICDcoding </h1>
This project is part of MS thesis conducted at AlgoAnalytics
<h2> Table of contents </h2>
<ul>
<li>ICD coding </li>
<li>MIMIC III dataset</li>
<li> Non-AI method </li>
 <ul>
 <li> Method and Algorithm </li>
 <li> Results </li>
 </ul>
 <li> Logistic Regression </li>
 <li> Deeplearning Model </li>
</ul>

<h2> About ICD coding </h2>
The International Classification of Diseases (ICD) is a globally used diagnostic tool for epidemiology,health
management and clinical purposes which lists different dieseases, injuries and procedures carried out in the
patients in a hierarchical manner. These codes are maintained, reglarly revised and maintained by WHO. <br>
These codes are necessary for maintaing electronic health records (EHR), billing and insurance claims and patient
management. Therfore correct assignment of these codes are not only important for patient health but also of
their economic importance.<br>
In today's ever growing health industry, the process should not be only accurate but fast. However the coding
process is very tedious and requires extensive training of the medical coders. Therefore the automatic prediction
of ICD codes from unstructured medical text would benefit human coders to save time, eliminate errors and
minimize costs. <br>
The latest revision of ICD codes are ICD10. The codes are hierarchical. Every code has a top level category
labelled as "ICD category" and a specifc code labelled as "ICD code". The goal of this project is to
assign ICD10 codes from the freely available de-identified patient records in the MIMIC III database. We plan
to study this task from NonAI to Deeplearning models.

<h2> MIMIC III dataset </h2>

> MIMIC III descriptive statistics
<table>
 <tr>
  <th>Dataset</th>
  <th>Hospital Admissions</th>
  <th>ICD9 codes</th>
  <th>ICD9 categories</th>
 </tr>
 <tr>
  <td> Full MIMIC III </td>
  <td> 58929 </td>
  <td> 6984 </td>
  <td> 1070 </td>
 </tr>
 <tr>
  <td> NOTEEVENTS </td>
  <td> 58329 </td>
  <td> 6967 </td>
  <td> 1070 </td>
 </tr>
 <tr>
  <td> Discharge Summaries </td>
  <td> 52722 </td>
  <td> 6919 </td>
  <td> 1069 </td>
 </tr>
</table>

> ICD code and category descriptive statistics
<table>
 <tr>
  <td>
   <table>
    <tr>
     <th>Frequency</th>
     <th>AdmIds</th>
     <th>%cover</th>
    </tr>
    <tr>
     <td>Top10</td>
     <td>40562</td>
     <td>76.93%</td>
    </tr>
    <tr>
     <td>Top20</td>
     <td>43958</td>
     <td>83.37%</td>
    </tr>
    <tr>
     <td>Top50</td>
     <td>49534</td>
     <td>93.95%</td>
    </tr>
    <tr>
     <td>Top100</td>
     <td>50625</td>
     <td>96.02%</td>
    </tr>
   </table>
  </td>
  <td>
   <table>
    <tr>
     <th>Frequency</th>
     <th>AdmIds</th>
     <th>%cover</th>
    </tr>
    <tr>
     <td>Top10</td>
     <td>44410</td>
     <td>84.42%</td>
    </tr>
    <tr>
     <td>Top20</td>
     <td>46089</td>
     <td>87.41%</td>
    </tr>
    <tr>
     <td>Top50</td>
     <td>49534</td>
     <td>96.55%</td>
    </tr>
    <tr>
     <td>Top100</td>
     <td>52007</td>
     <td>98.64%</td>
    </tr>
   </table>
  </td>
 </tr>
</table>
 

<table>
 <tr>
  <th> Dataset </th>
  <th> Hospital Admissions </th>
  <th> ICD9 PCS codes </th>
 </tr>
 <tr>
  <td> Full MIMIC III dataset </td>
  <td> 52243 </td>
  <td> 2009</td>
 </tr>
 <tr>
  <td> Discharge summaries </td>
  <td> 52726 </td>
  <td> 1989 </td>
 </tr>
</table>
<table>
 <tr>
  <th> Data </th>
  <th> Hospital Admissions </th>
  <th>  % coverage </th>
 </tr>
 <tr>
  <td> Top 10 ICD9 PCS codes </td>
  <td> 33304 </td>
  <td> 71.10 </td>
 </tr>
 <tr>
  <td> Top 20 ICD9 PCS codes </td>
  <td> 37931 </td>
  <td> 80.98 </td>
 </tr>
 <tr>
  <td> Top 50 ICD9 PCS codes </td>
  <td> 40493 </td>
  <td> 86.45</td>
 </tr>
</table>
