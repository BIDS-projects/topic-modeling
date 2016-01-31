"""
LDA Tests
"""

import pytest
from bidslda import bidslda
import arrow

class MockItem:
    def __init__(self, base_url, text):
        self.base_url = base_url
        self.text = text

@pytest.fixture
def mockitem():
    item = lambda x: x
    item.base_url = 'http://berkeley.edu'
    item.text = 'UC Berkeley rocks'
    return item

def test_lda():
    doc1 = MockItem("doc 1", "Skip to main content Intelligent research design for data intensive social science About About D-Lab Staff Dav Clark FAQ Contact Us Join Us Donate Services Training Past Trainings Consulting Working Groups Space Resources Data Resources Campus Resources Course List Blog & Events Blog Campus Events Calendar Intelligent research design for the age of data intensive social science. Helping social scientists collect, process, and visualize data Are you starting research or working on a project that uses data? Are you a data visualization expert looking for access to new data sets? D-Lab's collaborative environment caters to many types of data needs. The tools, methods, and techniques that D-Lab provides offer social scientists the ability to engage with complex research questions and produce answers that benefit academic colleagues, policymakers, and the public.")
    doc2 = MockItem("doc 2", "A flexible infrastructure of hardware, software, and above all human talent D-Lab is a new lab that aims to provide services, support, and a venue for research design and experimentation in data-intensive social sciences. Supporting research instruction wherever it occurs Researchers learn about new data, software and techniques in classrooms and lecture halls, but they also learn in online courses, through webinars, at personalized workshops, during seminars and brownbags, and through one-on-one consultations and discussions. D-Lab seeks to support those learning interactions, wherever and however they take place. Previous Pause Next Upcoming Trainings 14-Jan-16 10:00am INTENSIVE: QDA Day 4 - From Coding Qualitative Data to Analyzing It 14-Jan-16 12:30pm INTENSIVE: Stata 14-Jan-16 1:00pm INTENSIVE: R for Data Science Day 3 (analyzing data) See more Sign Up for Our Mailing List Keep up to date about the latest events, trainings, and news from the D-Lab! Email * Happy Holidays! The D-Lab will be closed December 21 thru January 1.")
    doc3 = MockItem("doc 3", "See our calendar for details of upcoming workshops. Blog The Season for Sharing Data: Working with the newly released Census 2010-2014 ACS 5 year data in R On December 3, 2015 the U.S. Census Bureau released the 2010-2014 5 year ACS (American Community Survey) data. Are you an R Hero? Join our Team! The D-Lab is hiring\xa0R\xa0instructors\xa0for the Spring semester to teach beginning and intermediate classes in data visualization and analysis! See more Featured Working Group Research in Practice Working Group So you\u2019ve got some of your graduate classes under your belt, and it's time to begin an original research project.\xa0 But how exactly are you going to go about surveying voters in Tanzania, interviewi TBD for Spring 2016, Kickoff meeting 12/3/15 at 4pm See more Featured Consultant Kunal Marwaha Python (Beginning, Data Cleaning/Preparation, Workflow Automation) Mondays 1-3pm, and by appointment I love to teach and consult for novice programmers, especially in Python: I can help if you are just getting started with programming. I could also be useful if your data currently lives in something like Excel and you want to analyze and visualize your data in a reproducible, efficient way. See more About | Contact | FAQ | Location | Work for Us D-Lab | University of California, Berkeley | 350 Barrows Hall Berkeley, CA 94720-3030 | dlab@berkeley.edu Connect with us Facebook Twitter RSS")
    corpus = [doc1.text, doc2.text, doc3.text]
    model = ldaimp.LDAM(2)
    f = open('outputs/log-' + str(arrow.now()) + '.txt', 'w')
    model.fit(corpus)
    for line in model.topics(2):
        f.write(str(line))
    f.close()

def test_input(mockitem):
    assert mockitem.base_url == 'http://berkeley.edu'
