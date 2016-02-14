# Python Script to Match Job Occupations (Textual Analysis)

[TOC]

## Installaion 

 1. install pip: `sudo apt-get install python-pip python-dev build-essential; sudo pip install --upgrade pip`
 2. Run **./install.sh** from the project directory

## Running Position Description Matching

 1. Add to **PYTHONPATH** the path to **site-aggregation** project, eg, `export PYTHONPATH=$HOME/site-aggregation`
 2. Set the MySQL user password either in the configuration of *site-aggregation* project or using **MYSQL_PWD**, eg, `export MYSQL_PWD=******`
 3. Run **./match_pd.py** script (it might take a few minutes to run it)

## Original Job Details

We need you to write a python script to match job occupations. This job involves a far amount of textual analysis. 

We have two files. Both have job occupations. Here the descriptions of the files:

 - **File 1**: It has job title, sometimes a job description, and sometimes the industry.
 - **File 2**: It always has the following fields: industry, occupation code, occupation name, occupation description, sample of job titles, and the five most important tasks in that job.

Your job if to write a python script that picks a job occupation from File 1 and finds the best match in File 2. The script should also a score on how good the job match is. Samples of the two files are attached to this job post.

When you apply for the job, you have to describe how you plan on tackling this problem. The job is posted for $150, but you can make higher bids, specially you have clever ways on making this matching very good.


