"""
NED University Admissions AI Assistant  -  Streamlit UI
--------------------------------------------------------
Run:   streamlit run app.py
Keys:  put GOOGLE_API_KEY and GROQ_API_KEY inside a .env file next to this file.
"""

import os
import re
import json
import time
import asyncio

import streamlit as st
from dotenv import load_dotenv

# Streamlit runs your script in a worker thread; Google's client needs an event loop there.
try:
    asyncio.get_running_loop()
except RuntimeError:
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

st.set_page_config(
    page_title="NED Admissions AI Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_dotenv()

from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_groq import ChatGroq


# =====================================================================
# 1) DATA  (copied exactly from the notebook - not modified)
# =====================================================================
policies = [
  {
    "category": "Admissions & Eligibility",
    "topic": "HSC Eligibility Criteria",
    "text": "A minimum of 60% aggregate marks in HSC (Pre-Engineering or equivalent) is required for Bachelor of Engineering (BE) and BS Computer Science programs."
  },
  {
    "category": "Admissions & Eligibility",
    "topic": "HSC Part-II Awaiting Candidates",
    "text": "Applicants appearing in HSC Part-II or equivalent examinations can apply provided they passed HSC Part-I or equivalent with at least 50% marks."
  },
  {
    "category": "Admissions & Eligibility",
    "topic": "DAE Diploma Eligibility",
    "text": "Candidates with a Diploma of Associate Engineer (DAE) from SBTE in a relevant field must have secured at least 60% marks within the last 3 years to apply against reserved DAE seats."
  },
  {
    "category": "Admissions & Eligibility",
    "topic": "Pre-Medical Student Eligibility",
    "text": "Pre-Medical students can apply for BS Biomedical Engineering and Bio-Engineering programs, and may apply for other specified BS programs subject to completing a non-credit deficiency math course."
  },
  {
    "category": "Admissions & Eligibility",
    "topic": "Program Duration",
    "text": "All undergraduate degree programs at NED University are 4 years (8 semesters) in duration, except Bachelor of Architecture (B.Arch) which is a 5-year program."
  },
  {
    "category": "Admissions & Eligibility",
    "topic": "ICS Student Eligibility",
    "text": "Candidates with Intermediate in Computer Science (Physics, Mathematics, Computer Science) are eligible for BS Computer Science, Software Engineering, Computational Finance, and selected engineering branches depending on subject combinations."
  },
  {
    "category": "Admissions & Eligibility",
    "topic": "O & A-Level Equivalence",
    "text": "O & A-Level candidates must obtain an Equivalence Certificate from the IBCC (Inter Board Committee of Chairmen) verifying minimum 60% equivalent marks for engineering tracks."
  },
  {
    "category": "Admissions & Eligibility",
    "topic": "Multiple Category Application",
    "text": "Applicants can select multiple eligible categories (such as Regular, Self-Finance, Reserved, Sponsor) in a single online application form without paying extra fees except for Self-Finance."
  },
  {
    "category": "Admissions & Eligibility",
    "topic": "HSC Mark Improvement Policy",
    "text": "Candidates must submit mark sheets that are complete and officially declared before the final deadline specified in the admission schedule."
  },
  {
    "category": "Admissions & Eligibility",
    "topic": "Age Limit Policy",
    "text": "There is no strict upper age limit for undergraduate programs provided the candidate meets academic recency guidelines."
  },
  {
    "category": "Entry Test & Syllabus",
    "topic": "Entry Test Format",
    "text": "The NED Pre-Admission Entry Test is a Computer-Based Test (CBT) consisting of 100 Multiple Choice Questions (MCQs) divided into 4 sections with 25 questions each."
  },
  {
    "category": "Entry Test & Syllabus",
    "topic": "Entry Test Duration",
    "text": "The total duration allowed for completing all 100 questions in the NED Pre-Admission Entry Test is 120 minutes (2 hours)."
  },
  {
    "category": "Entry Test & Syllabus",
    "topic": "Pre-Engineering Test Subjects",
    "text": "The four test sections for Pre-Engineering applicants are English, Mathematics, Physics, and Chemistry (25 marks each)."
  },
  {
    "category": "Entry Test & Syllabus",
    "topic": "Computer Science Test Subjects",
    "text": "The test sections for Computer Science applicants are English, Mathematics, Physics, and Computer Science."
  },
  {
    "category": "Entry Test & Syllabus",
    "topic": "Pre-Medical Test Subjects",
    "text": "The test sections for Pre-Medical applicants applying for biomedical tracks are English, Biology, Physics, and Chemistry."
  },
  {
    "category": "Entry Test & Syllabus",
    "topic": "Entry Test Passing Marks",
    "text": "Candidates must secure a minimum of 50% (50 marks out of 100) to pass the entry test and qualify for merit listing."
  },
  {
    "category": "Entry Test & Syllabus",
    "topic": "Negative Marking Policy",
    "text": "There is no negative marking for incorrect or unattempted questions in the NED Pre-Admission Entry Test."
  },
  {
    "category": "Entry Test & Syllabus",
    "topic": "Multiple Test Attempts",
    "text": "NED conducts the Pre-Admission Entry Test in two phases (Phase-I and Phase-II), and candidates can attempt both phases with the best score considered for merit."
  },
  {
    "category": "Entry Test & Syllabus",
    "topic": "SAT Score Exemption",
    "text": "Candidates scoring at least 800 in SAT-I and 1500 in SAT-II (Physics, Chemistry/CS, Mathematics) can apply for entry test exemption."
  },
  {
    "category": "Entry Test & Syllabus",
    "topic": "Calculator Allowance",
    "text": "Basic non-programmable scientific calculators are permitted inside the examination hall during the computer-based entry test."
  },
  {
    "category": "Merit Aggregate & Scoring",
    "topic": "Merit Formula",
    "text": "The aggregate merit score is calculated using 60% weightage for the NED Pre-Admission Entry Test marks and 40% weightage for HSC Part-I / O-Level marks."
  },
  {
    "category": "Merit Aggregate & Scoring",
    "topic": "HSC Component Calculation",
    "text": "The HSC component in merit aggregate is computed as (HSC Part-I Obtained Marks / Total HSC Part-I Marks) * 40."
  },
  {
    "category": "Merit Aggregate & Scoring",
    "topic": "Entry Test Component Calculation",
    "text": "The Entry Test component in merit aggregate is computed as (Entry Test Score / 100) * 60."
  },
  {
    "category": "Merit Aggregate & Scoring",
    "topic": "Merit List Publication",
    "text": "Official merit lists are published directly on the official NEDUET Admission Portal (www.neduet.edu.pk/admission)."
  },
  {
    "category": "Merit Aggregate & Scoring",
    "topic": "Merit Tie-Breaker Policy",
    "text": "In case of equal aggregate scores, marks obtained in specific entry test sections (such as Mathematics and Physics) followed by HSC marks serve as tie-breakers."
  },
  {
    "category": "Campus Navigation",
    "topic": "Transport & Bus Routes",
    "text": "NED University operates a wide fleet of point buses covering almost all major residential routes across Karachi for students and staff."
  },
  {
    "category": "Campus Navigation",
    "topic": "Point Bus Schedule",
    "text": "Point buses operate on fixed morning pickups to bring students to campus before 08:00 AM and offer afternoon return trips aligned with class dismissals."
  },
  {
    "category": "Campus Navigation",
    "topic": "Evening Program Transport",
    "text": "Designated point transport routes service postgraduate and evening program class schedules."
  },
  {
    "category": "Campus Navigation",
    "topic": "Transport Pass Registration",
    "text": "Students can register for transport passes through the Transport Office or Student Affairs Section by submitting the transport application along with prescribed fees."
  },
  {
    "category": "Campus Navigation",
    "topic": "Hostel Facilities",
    "text": "NED provides hostel accommodations for foreign students and non-resident students originating outside Karachi."
  },
  {
    "category": "Fee Structure & Refund Policies",
    "topic": "Initial Admission Fee",
    "text": "The baseline undergraduate admission fee ranges around PKR 70,000 to PKR 80,000 for the first semester, including admission and security deposits."
  },
  {
    "category": "Fee Structure & Refund Policies",
    "topic": "Self-Finance Upgrade Refund",
    "text": "If an applicant upgrades from a Self-Finance seat to a Regular seat, the paid Self-Finance fee is fully refunded upon submitting the required refund form."
  },
  {
    "category": "Fee Structure & Refund Policies",
    "topic": "Self-Finance Category Fee",
    "text": "Self-Finance category payments are one-time admission fees; regular semester tuition fees must still be paid separately every semester."
  },
  {
    "category": "Fee Structure & Refund Policies",
    "topic": "Admission Fee Payment Deadline",
    "text": "Admission status is not secured unless the required admission fee is paid on the day of the interview or admission call."
  },
  {
    "category": "Fee Structure & Refund Policies",
    "topic": "Accepted Payment Methods",
    "text": "Fees can be deposited via Pay Order / Demand Draft made to 'Director Finance, NEDUET' or via online banking portals at designated campus branches."
  },
  {
    "category": "Examinations & Document Verification",
    "topic": "Transcript Verification Process",
    "text": "Applicants must submit the official Verification Form to the Controller of Examinations section along with required document copies and fee."
  },
  {
    "category": "Examinations & Document Verification",
    "topic": "Transcript Processing Timeline",
    "text": "Standard transcript processing takes 15 working days, while urgent transcript requests take approximately 10 working days."
  },
  {
    "category": "Examinations & Document Verification",
    "topic": "Duplicate Degree Issuance",
    "text": "Duplicate degree issuance requires an affidavit on Rs. 50 stamp paper, an undertaking on Rs. 100 stamp paper, a copy of the lost degree newspaper ad, CNIC, and prescribed fees."
  },
  {
    "category": "Examinations & Document Verification",
    "topic": "Online & Courier Verification",
    "text": "Document verification requests can be submitted remotely via email or courier with fee payment deposited directly to designated university accounts."
  },
  {
    "category": "Examinations & Document Verification",
    "topic": "Answer Sheet Scrutiny Deadline",
    "text": "Scrutiny applications for semester examination papers must be submitted within 15 days of result declaration."
  },
  {
    "category": "Campus Procedures & Miscellaneous",
    "topic": "Required Documents for Interview",
    "text": "Candidates must present their Admit Card, original HSC/SSC mark sheets, domicile, PRC, CNIC/B-Form, passport photographs, and signed Annexure forms."
  },
  {
    "category": "Campus Procedures & Miscellaneous",
    "topic": "Main Campus Location",
    "text": "NED University Main Campus is situated on Main University Road, adjacent to the University of Karachi in Gulshan-e-Iqbal, Karachi."
  },
  {
    "category": "Campus Procedures & Miscellaneous",
    "topic": "Sub-Campus Location",
    "text": "NED operates a sub-campus, the Thar Institute of Engineering, Sciences and Technology (TIEST), located in Mithi, Tharparkar."
  },
  {
    "category": "Campus Procedures & Miscellaneous",
    "topic": "Deficiency Mathematics Course",
    "text": "Students admitted without intermediate mathematics background must pass non-credit deficiency mathematics modules during their first semester."
  },
  {
    "category": "Campus Procedures & Miscellaneous",
    "topic": "Sports Facilities",
    "text": "NED campus houses extensive outdoor sports grounds, indoor sports complexes, a gymnasium, and swimming facilities."
  },
  {
    "category": "Campus Procedures & Miscellaneous",
    "topic": "Admit Card Generation",
    "text": "Entry test admit cards are generated digitally on the official NED Admission Portal once the application and payment voucher are verified."
  },
  {
    "category": "Campus Procedures & Miscellaneous",
    "topic": "Sports Quota Seats",
    "text": "Reserved sports quota seats are allocated based on athletic achievements and trials conducted by the university sports board."
  },
  {
    "category": "Campus Procedures & Miscellaneous",
    "topic": "Semester Academic Standing",
    "text": "Undergraduate students must maintain a minimum cumulative GPA (CGPA) of 2.00 to stay in good academic standing."
  },
  {
    "category": "Campus Procedures & Miscellaneous",
    "topic": "Scholarships & Financial Aid",
    "text": "Financial aid including HEC Need-Based Scholarships, Merit Scholarships, and alumni grants are administered by the Financial Assistance Office."
  },
  {
    "category": "Campus Procedures & Miscellaneous",
    "topic": "Admission Contact Information",
    "text": "Candidates can contact the Admission Office via telephone (+92-21-99261261-8 Ext: 2412) or official inquiry email on the website."
  },
  {
    "category": "Faculties & Academic Departments",
    "topic": "Engineering Faculties Structure",
    "text": "NED University comprises five primary faculties: Civil & Petroleum Engineering, Mechanical & Manufacturing Engineering, Electrical & Computer Engineering, Chemical & Process Engineering, and Architecture & Sciences."
  },
  {
    "category": "Faculties & Academic Departments",
    "topic": "Software & CS Specializations",
    "text": "The Department of Software Engineering and Computer Science offers specialized tracks in Artificial Intelligence, Data Science, Cyber Security, and Gaming/Animation."
  },
  {
    "category": "Faculties & Academic Departments",
    "topic": "Civil Engineering Specializations",
    "text": "The Department of Civil Engineering provides specialized degree tracks in Urban Engineering, Construction Engineering, and Petroleum Engineering alongside general Civil Engineering."
  },
  {
    "category": "Faculties & Academic Departments",
    "topic": "Textile Engineering Programs",
    "text": "The Department of Textile Engineering offers a 4-year Bachelor of Engineering (BE) in Textile Engineering as well as a BS in Textile Sciences."
  },
  {
    "category": "Faculties & Academic Departments",
    "topic": "Architecture & Planning Degrees",
    "text": "The Department of Architecture and Planning offers a 5-year professional Bachelor of Architecture (B.Arch) degree and Urban and Regional Planning tracks."
  },
  {
    "category": "Faculties & Academic Departments",
    "topic": "Food Engineering Program",
    "text": "The Department of Food Engineering under the Faculty of Chemical & Process Engineering offers a 4-year Bachelor of Engineering in Food Engineering."
  },
  {
    "category": "Faculties & Academic Departments",
    "topic": "Science & Humanities BS Degrees",
    "text": "The Faculty of Architecture and Sciences offers BS degrees in Physics, Chemistry, Applied Mathematics, Computational Finance, Economics & Finance, Management Sciences, and English Linguistics."
  },
  {
    "category": "Faculties & Academic Departments",
    "topic": "TIEST Degree Programs",
    "text": "The Thar Institute of Engineering, Sciences and Technology (TIEST) offers undergraduate degrees in BE Civil Engineering, BS Computer Science, BS Management Sciences, and BS English."
  },
  {
    "category": "Faculties & Academic Departments",
    "topic": "Biomedical Engineering Department",
    "text": "Biomedical Engineering is operated under the Faculty of Electrical and Computer Engineering via the Department of Bio-Medical Engineering."
  },
  {
    "category": "Faculties & Academic Departments",
    "topic": "Automotive & Marine Engineering",
    "text": "The Department of Automotive and Marine Engineering offers a specialized 4-year Bachelor of Engineering in Automotive Engineering."
  },
  {
    "category": "Postgraduate & Research Programs",
    "topic": "Postgraduate Degree Options",
    "text": "NED offers Master of Engineering (M.Engg.), Master of Science (MS), Master of Business Administration (MBA), and Doctor of Philosophy (Ph.D.) degrees."
  },
  {
    "category": "Postgraduate & Research Programs",
    "topic": "Master Degree Duration",
    "text": "Most M.Engg. and MS programs are designed as 2.5-year evening or weekend structured programs for working professionals."
  },
  {
    "category": "Postgraduate & Research Programs",
    "topic": "Advanced Research Board (ASRB)",
    "text": "The Advanced Studies and Research Board (ASRB) oversees doctoral studies, research thesis evaluations, and advisor assignments."
  },
  {
    "category": "Postgraduate & Research Programs",
    "topic": "Graduate Admission Test",
    "text": "Postgraduate admission requires clearing the NED Graduate Entry Test or submitting valid scores from NTS-GAT or HEC HAT tests."
  },
  {
    "category": "Postgraduate & Research Programs",
    "topic": "Engineering Management Programs",
    "text": "NED offers Master of Engineering Management (MEM) programs specializing in Industrial Management, Quality Management, and Supply Chain Management."
  },
  {
    "category": "Postgraduate & Research Programs",
    "topic": "Water Research Institute",
    "text": "The Panjwani-Hisar Water Institute (PHWI) conducts research in Urban Water Management, Climate Change Management, and sustainable water systems."
  },
  {
    "category": "Postgraduate & Research Programs",
    "topic": "Direct Ph.D. Policy",
    "text": "High-merit undergraduate degree holders meeting strict CGPA thresholds can apply for direct Ph.D. enrollment per HEC and university guidelines."
  },
  {
    "category": "Postgraduate & Research Programs",
    "topic": "Postgraduate Class Timing",
    "text": "Postgraduate and Master degree lectures are conducted during evening hours or weekends to accommodate full-time working professionals."
  },
  {
    "category": "Postgraduate & Research Programs",
    "topic": "Postgraduate CGPA Threshold",
    "text": "Postgraduate students must maintain a minimum cumulative GPA of 2.50 to stay enrolled in coursework and thesis phases."
  },
  {
    "category": "Postgraduate & Research Programs",
    "topic": "Official Research Journal",
    "text": "NED University publishes the peer-reviewed 'NED University Journal of Research' focusing on applied sciences and engineering technology."
  },
  {
    "category": "Historical Background & Milestones",
    "topic": "Establishment Year",
    "text": "The institution was founded in 1922 as NED Government Engineering College, making it the oldest engineering education institute in Pakistan."
  },
  {
    "category": "Historical Background & Milestones",
    "topic": "University Chartering",
    "text": "The college was fully upgraded and chartered as a university on March 1, 1977."
  },
  {
    "category": "Historical Background & Milestones",
    "topic": "Institutional Name Origin",
    "text": "The university is named after philanthropist Nadirshaw Edulji Dinshaw, whose family funded the initial establishment of the college."
  },
  {
    "category": "Historical Background & Milestones",
    "topic": "Original Campus Location",
    "text": "The original campus was located near DJ Science College before moving to its current main campus location on Main University Road in the 1970s."
  },
  {
    "category": "Historical Background & Milestones",
    "topic": "Centenary Milestone",
    "text": "NED University has completed over 100 years of technical and engineering education in Pakistan."
  },
  {
    "category": "Student Life & Societies",
    "topic": "Active Campus Societies",
    "text": "Over 40 active student societies operate on campus including IEEE NED, ASME, IMechE, ACM, Debating Society, and Literary Society."
  },
  {
    "category": "Student Life & Societies",
    "topic": "Motorsport SAE Team",
    "text": "Student groups design and construct formula-style racing vehicles to compete in national and global engineering design competitions."
  },
  {
    "category": "Student Life & Societies",
    "topic": "ORIC Startup Support",
    "text": "The Office of Research, Innovation and Commercialization (ORIC) assists students with prototype commercialization, patent filings, and industry links."
  },
  {
    "category": "Student Life & Societies",
    "topic": "Startup Incubation Center",
    "text": "The National Incubation Center Karachi (NICK) collaborates with NED ORIC to mentor student tech entrepreneurs and startups."
  },
  {
    "category": "Student Life & Societies",
    "topic": "Annual Campus Events",
    "text": "Major annual campus events include the NED Olympiad, Procom tech festival, and intra-university sports tournaments."
  },
  {
    "category": "Student Life & Societies",
    "topic": "Student Publications",
    "text": "Various departments and student society chapters publish technical newsletters, research magazines, and blogs."
  },
  {
    "category": "Student Life & Societies",
    "topic": "Outdoor Adventure Club",
    "text": "The NED Adventure Club organizes trekking, hiking, and northern area camping trips for students."
  },
  {
    "category": "Student Life & Societies",
    "topic": "NED Alumni Network (NEDIANs)",
    "text": "The global NEDIANs Alumni Network provides scholarship endowments, student mentorship, campus infrastructure funds, and networking support."
  },
  {
    "category": "Student Life & Societies",
    "topic": "Student Affairs Directorate",
    "text": "The Directorate of Student Affairs oversees student society activities, character-building programs, and event approvals."
  },
  {
    "category": "Student Life & Societies",
    "topic": "Arts & Cultural Societies",
    "text": "Student-managed music, dramatics, and fine arts societies host variety shows, qawwali nights, and dramatic stage performances."
  },
  {
    "category": "Library & Campus Services",
    "topic": "Central Library Facility",
    "text": "The Central Library houses technical reference collections, printed journals, digital archive stations, and individual study cubicles."
  },
  {
    "category": "Library & Campus Services",
    "topic": "HEC Digital Library Access",
    "text": "Students are granted access to the HEC Digital Library database containing full-text papers from IEEE, Elsevier, Springer, and ACM."
  },
  {
    "category": "Library & Campus Services",
    "topic": "Campus Wi-Fi Connectivity",
    "text": "The entire campus is covered with high-speed fiber internet infrastructure and wireless Eduroam network access."
  },
  {
    "category": "Library & Campus Services",
    "topic": "Civil Engineering Testing Labs",
    "text": "The Civil Department maintains advanced testing facilities for concrete durability, seismic simulation, soil dynamics, and structural analysis."
  },
  {
    "category": "Library & Campus Services",
    "topic": "High-Performance Computing Hubs",
    "text": "Computer Science labs feature high-end GPU workstations dedicated to AI, machine learning, and data analytics computation."
  },
  {
    "category": "Library & Campus Services",
    "topic": "Mechanical Engineering Workshops",
    "text": "Mechanical facilities include machine shops, casting yards, CNC machining centers, and thermal power testing laboratories."
  },
  {
    "category": "Library & Campus Services",
    "topic": "Learning Management System (LMS)",
    "text": "Students use the online NED LMS portal for course materials, assignment submissions, class attendance tracking, and grade updates."
  },
  {
    "category": "Library & Campus Services",
    "topic": "Campus Health Clinic",
    "text": "An on-campus medical dispensary staffed with qualified doctors provides emergency first-aid and routine health consultations."
  },
  {
    "category": "Library & Campus Services",
    "topic": "Banking & ATM Services",
    "text": "Multiple commercial bank branches and 24/7 ATM facilities operate within the main campus grounds."
  },
  {
    "category": "Library & Campus Services",
    "topic": "Cafeterias & Food Outlets",
    "text": "Multiple cafeterias and tea stalls across faculty blocks offer hygienic food options at subsidized rates."
  },
  {
    "category": "Library & Campus Services",
    "topic": "Student Parking Facilities",
    "text": "Designated car and motorcycle parking lots with security supervision are available inside campus gates."
  },
  {
    "category": "Library & Campus Services",
    "topic": "Campus Security Network",
    "text": "Campus security is enforced through perimeter walls, CCTV surveillance, automated boom barriers, and security personnel."
  },
  {
    "category": "Library & Campus Services",
    "topic": "Career Placement Bureau",
    "text": "The Career Guidance and Placement Bureau (CGPB) organizes job fairs, corporate interviews, and resume-building workshops."
  },
  {
    "category": "Library & Campus Services",
    "topic": "Solar Energy Sustainability",
    "text": "NED has installed rooftop solar panel arrays across major campus buildings as part of its green energy transition."
  },
  {
    "category": "Library & Campus Services",
    "topic": "Official Directory Contact",
    "text": "Complete departmental extension directories and official contacts are published on the website www.neduet.edu.pk."
  },
  {
    "category": "Admissions & Eligibility",
    "topic": "Hafiz-e-Quran Weightage",
    "text": "Certified Hafiz-e-Quran applicants who pass the entry test receive 20 additional marks added to their academic aggregate pool upon Sanad verification."
  },
  {
    "category": "Admissions & Eligibility",
    "topic": "Multiple Exam Boards Policy",
    "text": "Candidates who passed HSC Part-I from one exam board and HSC Part-II from a different board are not eligible for admission."
  },
  {
    "category": "Admissions & Eligibility",
    "topic": "Re-admission for Enrolled Students",
    "text": "Currently enrolled NED students cannot apply for fresh admission unless their existing university enrollment is officially cancelled."
  },
  {
    "category": "Admissions & Eligibility",
    "topic": "Quota & Domicile Policy",
    "text": "NED University admissions strictly adhere to urban and rural Sindh domicile quotas, verified through official prospectus guidelines."
  },
  {
    "category": "Admissions & Eligibility",
    "topic": "HSC Passing Recency Rule",
    "text": "The year of passing the intermediate or equivalent exam must not be earlier than three years prior to the application submission date."
  },
  {
    "category": "Admissions & Eligibility",
    "topic": "Standardized SAT Alternatives",
    "text": "International applicants can substitute entry tests with SAT scores: minimum SAT-I score of 800 and SAT-II score of 1500 in Math, Physics, and Chemistry/CS."
  },
  {
    "category": "Admissions & Eligibility",
    "topic": "Provisional Admission Verification",
    "text": "All candidate admissions remain strictly provisional until original HSC mark sheets are fully verified by the university registrar."
  },
  {
    "category": "Admissions & Eligibility",
    "topic": "Attested Application Affidavits",
    "text": "Applicants must submit printed online forms and required conduct affidavits attested by an Oath Commissioner at the time of interview."
  },
  {
    "category": "Admissions & Eligibility",
    "topic": "Awaiting Result Eligibility",
    "text": "Candidates awaiting HSC Part-II results are permitted to apply conditionally based on their declared HSC Part-I results."
  },
  {
    "category": "Admissions & Eligibility",
    "topic": "Entry Test Score Validity",
    "text": "NED Entry Test scores are valid only for the current academic session and cannot be carried forward to future years."
  },
  {
    "category": "Faculties & Academic Departments",
    "topic": "Materials & Metallurgical Engineering",
    "text": "The Department of Metallurgical and Materials Engineering offers separate BE degrees in Materials Engineering and Metallurgical Engineering."
  },
  {
    "category": "Faculties & Academic Departments",
    "topic": "Polymer & Petrochemical Engineering",
    "text": "The Department of Polymer and Petrochemical Engineering under the Chemical Faculty administers polymer engineering degree tracks."
  },
  {
    "category": "Faculties & Academic Departments",
    "topic": "Software Elective Domains",
    "text": "Software Engineering students can specialize in elective streams including Artificial Intelligence, Data Science, Cyber Security, and Gaming."
  },
  {
    "category": "Faculties & Academic Departments",
    "topic": "Earthquake Engineering Center",
    "text": "The Department of Civil Engineering operates dedicated research wings including an Earthquake Engineering Center and Environmental Engineering lab."
  },
  {
    "category": "Faculties & Academic Departments",
    "topic": "Electronic Engineering Programs",
    "text": "The Department of Electronic Engineering offers a 4-year BE along with postgraduate M.Engg tracks in Industrial Electronics and IC Design."
  },
  {
    "category": "Faculties & Academic Departments",
    "topic": "Telecommunications Engineering",
    "text": "Telecommunications research focuses on optical fiber communications, wireless network architectures, mobile protocols, and signal processing."
  },
  {
    "category": "Faculties & Academic Departments",
    "topic": "Computer Systems Engineering",
    "text": "The Computer and Information Systems Engineering department focuses on embedded systems, hardware architecture, and computer networking."
  },
  {
    "category": "Faculties & Academic Departments",
    "topic": "Urban & Infrastructure Engineering",
    "text": "The Department of Urban and Infrastructure Engineering provides civil engineering education tailored for municipal systems and urban transit."
  },
  {
    "category": "Faculties & Academic Departments",
    "topic": "Essential Studies Department",
    "text": "The Department of Essential Studies delivers core general education courses including Islamic Studies, Pakistan Studies, and Professional Ethics."
  },
  {
    "category": "Faculties & Academic Departments",
    "topic": "Applied Sciences BS Programs",
    "text": "Applied science options include four-year BS degrees in Computational Finance, Applied Mathematics, Industrial Chemistry, and Physics."
  },
  {
    "category": "Accreditation & Professional Bodies",
    "topic": "PEC Engineering Accreditation",
    "text": "All Bachelor of Engineering (BE) programs offered by NED University are officially accredited by the Pakistan Engineering Council (PEC)."
  },
  {
    "category": "Accreditation & Professional Bodies",
    "topic": "PCATP Architecture Accreditation",
    "text": "The Bachelor of Architecture (B.Arch) program is certified and accredited by the Pakistan Council of Architects and Town Planners (PCATP)."
  },
  {
    "category": "Accreditation & Professional Bodies",
    "topic": "HEC Degree Recognition",
    "text": "All non-engineering BS, MS, MBA, and Ph.D. degree programs are fully recognized by the Higher Education Commission (HEC) of Pakistan."
  },
  {
    "category": "Accreditation & Professional Bodies",
    "topic": "HEC QEC Ranking",
    "text": "NED's Quality Enhancement Cell (QEC) is consistently placed in the top 'W' performance category in official HEC evaluations."
  },
  {
    "category": "Accreditation & Professional Bodies",
    "topic": "ISO Quality Certification",
    "text": "NED University maintains official ISO 9001:2015 quality management system certification across its academic and administrative units."
  },
  {
    "category": "Examinations & Grading",
    "topic": "Maximum Degree Duration",
    "text": "Students must fulfill all graduation requirements for an undergraduate engineering degree within a maximum time limit of 7 years."
  },
  {
    "category": "Examinations & Grading",
    "topic": "Course Registration Timeline",
    "text": "Semester course registration forms must be completed within two weeks of the official declaration of prior semester results."
  },
  {
    "category": "Examinations & Grading",
    "topic": "Examination Controller Office",
    "text": "The Controller of Examinations regulates end-term exam schedules, invigilation guidelines, grade cards, and official transcripts."
  },
  {
    "category": "Examinations & Grading",
    "topic": "Digital Registration Portal",
    "text": "Students submit semester course registrations through the online student portal without requiring paper filings."
  },
  {
    "category": "Examinations & Grading",
    "topic": "Student Academic History Portal",
    "text": "Enrolled students can log into their online portal 24/7 to inspect course records, GPA logs, and semester attendance percentages."
  },
  {
    "category": "Campus Infrastructure & Operations",
    "topic": "PERN Data Infrastructure",
    "text": "The university is linked to the Pakistan Education and Research Network (PERN) to provide high-bandwidth connectivity for research labs."
  },
  {
    "category": "Campus Infrastructure & Operations",
    "topic": "Eduroam Global Wi-Fi",
    "text": "The campus network supports Eduroam and EduGAIN federated roaming for visiting international scholars and students."
  },
  {
    "category": "Campus Infrastructure & Operations",
    "topic": "NED Academy Continuing Education",
    "text": "NED Academy acts as the professional training wing, offering post-graduate diplomas, IT certificates, and executive training."
  },
  {
    "category": "Campus Infrastructure & Operations",
    "topic": "Carbon Neutrality Initiatives",
    "text": "The university implements green initiatives including rooftop solar generation, water recycling plants, and urban forestation drives."
  },
  {
    "category": "Campus Infrastructure & Operations",
    "topic": "Community Welfare Projects",
    "text": "NED operates community social projects including the Cowasjee Primary School and clean drinking water plants for surrounding areas."
  },
  {
    "category": "Campus Infrastructure & Operations",
    "topic": "Campus Daycare Center",
    "text": "An on-campus Daycare Center provides supervised childcare for university staff, faculty, and studying mothers."
  },
  {
    "category": "Campus Infrastructure & Operations",
    "topic": "Blood Donation & Welfare Drives",
    "text": "Student welfare societies regularly organize blood drive camps in partnership with national health foundations."
  },
  {
    "category": "Campus Infrastructure & Operations",
    "topic": "Digital Video Conferencing",
    "text": "NED maintains dedicated server facilities for video conferencing, virtual lectures, and Google Workspace integrations."
  },
  {
    "category": "Campus Infrastructure & Operations",
    "topic": "Foreign Language Certifications",
    "text": "NED Academy conducts language training certifications, including Mandarin Chinese language courses for engineering students."
  },
  {
    "category": "Campus Infrastructure & Operations",
    "topic": "International Linkages Office",
    "text": "The Directorate of Outreach & Networking manages foreign academic collaborations, international visitor programs, and student exchange setups."
  },
  {
    "category": "Campus Infrastructure & Operations",
    "topic": "One Window Clearance Desk",
    "text": "The One Window service center accelerates student administrative requests, transcript issuance, and fee clearances."
  },
  {
    "category": "Campus Infrastructure & Operations",
    "topic": "Infrastructure Grievance System",
    "text": "Students can log maintenance and facility complaints online through the Student Affairs portal or focal officers."
  },
  {
    "category": "Campus Infrastructure & Operations",
    "topic": "Licensing Exam Preparation",
    "text": "NED Academy offers preparatory coaching for engineering professional licensing exams and competitive tests like GAT and GRE."
  },
  {
    "category": "Campus Infrastructure & Operations",
    "topic": "Library Research Cubicles",
    "text": "The Central Library provides private research cubicles and dedicated reading zones for thesis researchers and graduate students."
  },
  {
    "category": "Campus Infrastructure & Operations",
    "topic": "Laboratory Safety Rules",
    "text": "Mandatory safety protocols, safety gear, eye-wash stations, and chemical disposal procedures are enforced in all laboratory areas."
  },
  {
    "category": "Campus Infrastructure & Operations",
    "topic": "Industrial Internship Coordination",
    "text": "The Placement Bureau facilitates mandatory 4-to-8 week summer industrial internships for senior engineering students."
  },
  {
    "category": "Campus Infrastructure & Operations",
    "topic": "Female Sports Arenas",
    "text": "Separate indoor athletic facilities, badminton courts, and dedicated gymnasium hours are reserved for female students."
  },
  {
    "category": "Campus Infrastructure & Operations",
    "topic": "Emergency Response System",
    "text": "Emergency call points and mobile security patrols operate across all academic sectors, parking lots, and walkways."
  },
  {
    "category": "Campus Infrastructure & Operations",
    "topic": "Industrial Testing Services",
    "text": "NED provides material testing, structural audit, and chemical analysis services to commercial firms through industrial testing cells."
  },
  {
    "category": "Campus Infrastructure & Operations",
    "topic": "Real-Time Admission Portal",
    "text": "Real-time updates regarding test seats, interview schedules, and merit lists are published on www.neduet.edu.pk."
  }
]


# =====================================================================
# 2) SYSTEM RULES  (copied exactly from the notebook - not modified)
# =====================================================================
NED_SYSTEM_RULES = """You are the proud, official NEDUET Admissions AI Assistant, dedicated to guiding prospective students on their journey to becoming part of our historic institution (established in 1922).

Your tone must be welcoming, encouraging, highly professional, and distinctly tailored for future 'NEDians'.

CRITICAL OPERATION RULES:
1. Always utilize the `search_ned_admission_rules` tool to pull concrete facts before answering. Treat returned documents as your absolute, sole source of truth.
2. Do not assume, generalize, or extrapolate policies (e.g., if eligibility for a specific engineering branch isn't stated for a background, do not guess it).
3. If no matching information exists in the retrieved database context, explicitly state that the available database policies do not specify this. Direct them to contact our official Admission Office via telephone (+92-21-99261261-8 Ext: 2412) or the official inquiry email.

PERSONALITY & STYLE INSTRUCTIONS:
- **Greeting & Close**: Greet the user warmly and sign off with an encouraging phrase like "Best of luck on your journey to becoming a NEDian!" or similar university-proud sentiments.
- **Formatting**: Structure information clearly using markdown tables, bold key headers, and bullet points. Never present long, unformatted blocks of text.
- **Evidence & Verification**: Always include a short "[Verified Source: <Topic Name>]" citation note under each claim so the applicant knows the information is officially verified.
"""


# =====================================================================
# 3) THEME  (light / dark)
# =====================================================================
LIGHT = {
    "bg": "#F2F5FB", "surface": "#FFFFFF", "soft": "#E9EEF9", "ink": "#101828",
    "muted": "#5B6577", "primary": "#1F3FBF", "primary_ink": "#FFFFFF",
    "gold": "#F5B50A", "border": "#D9E0EE", "user_bg": "#E4EBFF",
    "hero_a": "#1F3FBF", "hero_b": "#2B58E8", "shadow": "0 6px 24px rgba(31,63,191,.10)",
}
DARK = {
    "bg": "#0B1020", "surface": "#131A2E", "soft": "#1A2340", "ink": "#E8EDF8",
    "muted": "#9AA6BF", "primary": "#7C98FF", "primary_ink": "#0B1020",
    "gold": "#F5C542", "border": "#243049", "user_bg": "#1B2750",
    "hero_a": "#14214F", "hero_b": "#1E3378", "shadow": "0 6px 24px rgba(0,0,0,.45)",
}

CSS_TEMPLATE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

.stApp { background: %%bg%%; }
.stApp p, .stApp li, .stApp label, .stApp td, .stApp th, .stApp h1, .stApp h2, .stApp h3, .stApp h4,
.stApp strong, .stApp em, .stApp del, .stApp summary, .stApp textarea {
    font-family: 'Plus Jakarta Sans', system-ui, -apple-system, 'Segoe UI', sans-serif;
}
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] strong, [data-testid="stMarkdownContainer"] em,
[data-testid="stMarkdownContainer"] del, [data-testid="stMarkdownContainer"] td,
[data-testid="stMarkdownContainer"] h1, [data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3, [data-testid="stMarkdownContainer"] h4 { color: %%ink%% !important; }
[data-testid="stMarkdownContainer"] th { color: %%ink%% !important; background: %%soft%% !important; }
[data-testid="stMarkdownContainer"] a { color: %%primary%% !important; }
[data-testid="stMarkdownContainer"] code { background: %%soft%%; color: %%ink%%; border-radius: 6px; }
[data-testid="stMarkdownContainer"] table { border: 1px solid %%border%%; border-radius: 10px; border-collapse: separate; border-spacing: 0; overflow: hidden; }
[data-testid="stMarkdownContainer"] td, [data-testid="stMarkdownContainer"] th { border-color: %%border%% !important; }
[data-testid="stMarkdownContainer"] li::marker { color: %%gold%%; }

/* Header / chrome */
[data-testid="stHeader"] { background: transparent; }
[data-testid="stToolbar"], [data-testid="stDecoration"], #MainMenu, footer { display: none !important; }
.block-container { padding-top: 1.6rem; padding-bottom: 7rem; max-width: 980px; }

/* Sidebar */
[data-testid="stSidebar"] { background: %%surface%%; border-right: 1px solid %%border%%; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: %%ink%% !important; }
.side-brand { display:flex; align-items:center; gap:.7rem; margin-bottom:.4rem; }
.side-brand .logo { width:42px; height:42px; border-radius:12px; background:%%primary%%; color:%%primary_ink%%;
    display:flex; align-items:center; justify-content:center; font-size:1.4rem; box-shadow:%%shadow%%; }
.side-brand .name { font-family:'Fraunces', Georgia, serif; font-weight:700; font-size:1.15rem; color:%%ink%%; line-height:1.1; }
.side-brand .tag { font-size:.75rem; color:%%muted%%; }
.contact-card { background:%%soft%%; border:1px solid %%border%%; border-radius:14px; padding:.85rem 1rem; font-size:.85rem; color:%%ink%%; line-height:1.55; }
.contact-card b { color:%%ink%%; }
.contact-card a { color:%%primary%%; text-decoration:none; font-weight:600; }
.side-note { font-size:.74rem; color:%%muted%%; line-height:1.5; }

/* Expanders */
[data-testid="stExpander"] { border:1px solid %%border%%; border-radius:14px; background:%%surface%%; }
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary p, [data-testid="stExpander"] summary span { color:%%ink%% !important; }

/* Hero */
.hero { background: linear-gradient(135deg, %%hero_a%% 0%, %%hero_b%% 100%); border-radius: 22px; padding: 2.2rem 2.2rem 2rem 2.2rem;
    position: relative; overflow: hidden; box-shadow: %%shadow%%; border: 1px solid %%border%%; margin-bottom: 1.2rem; }
.hero::before { content:""; position:absolute; left:0; top:0; bottom:0; width:7px; background:%%gold%%; }
.hero .est { font-family:'Fraunces', Georgia, serif; font-weight:700; font-size:7.5rem; line-height:1; color:rgba(255,255,255,.10);
    position:absolute; right:1.4rem; bottom:-1.1rem; letter-spacing:-2px; user-select:none; }
.hero h1 { font-family:'Fraunces', Georgia, serif; font-weight:700; font-size:2.35rem; line-height:1.12; margin:0 0 .6rem 0;
    color:#FFFFFF !important; letter-spacing:-.5px; }
.hero p { color:rgba(255,255,255,.88) !important; font-size:1.02rem; line-height:1.6; max-width:34rem; margin:0; }
.hero .pills { margin-top:1.1rem; display:flex; flex-wrap:wrap; gap:.5rem; }
.hero .pill { background:rgba(255,255,255,.14); border:1px solid rgba(255,255,255,.25); color:#fff; border-radius:999px;
    padding:.28rem .8rem; font-size:.8rem; font-weight:600; }

/* Section title */
.section-title { font-family:'Fraunces', Georgia, serif; font-weight:600; font-size:1.15rem; color:%%ink%%; margin:1.2rem 0 .6rem 0; }

/* Buttons (quick questions, topics) */
.stButton > button { background:%%surface%%; border:1px solid %%border%%; border-radius:14px; padding:.75rem 1rem;
    text-align:left; justify-content:flex-start; transition: border-color .15s, background .15s, transform .15s; width:100%; }
.stButton > button p, .stButton > button span { color:%%ink%% !important; font-weight:500; font-size:.92rem; text-align:left; }
.stButton > button:hover { border-color:%%primary%%; background:%%soft%%; transform: translateY(-1px); }
.stButton > button:focus-visible { outline:2px solid %%primary%%; outline-offset:2px; }

/* Chat messages */
[data-testid="stChatMessage"] { background:%%surface%%; border:1px solid %%border%%; border-radius:18px; padding:1rem 1.1rem; margin-bottom:.8rem; box-shadow:%%shadow%%; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]),
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) { background:%%user_bg%%; }
[data-testid="stChatMessage"] table { width:100%; font-size:.92rem; }

/* Chat input */
[data-testid="stBottom"], [data-testid="stBottom"] > div { background:%%bg%% !important; }
[data-testid="stChatInput"], [data-testid="stChatInput"] > div { background:%%surface%% !important; }
[data-testid="stChatInput"] { border:1.5px solid %%border%%; border-radius:16px; box-shadow:%%shadow%%; }
[data-testid="stChatInput"]:focus-within { border-color:%%primary%%; }
[data-testid="stChatInput"] textarea { color:%%ink%% !important; background:transparent !important; }
[data-testid="stChatInput"] textarea::placeholder { color:%%muted%% !important; opacity:1; }
[data-testid="stChatInput"] button { background:%%primary%% !important; color:%%primary_ink%% !important; border-radius:10px; }
[data-testid="stChatInput"] button svg { fill:%%primary_ink%% !important; color:%%primary_ink%% !important; }

/* Spinner / status / alerts */
[data-testid="stSpinner"] p, [data-testid="stStatusWidget"] { color:%%ink%% !important; }
[data-testid="stStatus"] { background:%%surface%%; border:1px solid %%border%%; border-radius:14px; }
[data-testid="stStatus"] p, [data-testid="stStatus"] summary span { color:%%ink%% !important; }

@media (max-width: 640px) {
    .hero { padding: 1.5rem 1.2rem 1.4rem 1.4rem; }
    .hero h1 { font-size: 1.7rem; }
    .hero .est { font-size: 4.5rem; }
}
@media (prefers-reduced-motion: reduce) {
    .stButton > button { transition: none; transform: none !important; }
}
</style>
"""


def build_css(dark: bool) -> str:
    palette = DARK if dark else LIGHT
    css = CSS_TEMPLATE
    for key, value in palette.items():
        css = css.replace(f"%%{key}%%", value)
    return css


# =====================================================================
# 4) SESSION STATE
# =====================================================================
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending" not in st.session_state:
    st.session_state.pending = None

st.markdown(build_css(st.session_state.dark_mode), unsafe_allow_html=True)


def set_pending(question: str):
    st.session_state.pending = question


def clear_chat():
    st.session_state.messages = []
    st.session_state.pending = None


# =====================================================================
# 5) API KEYS  (from .env)
# =====================================================================
google_key = (os.getenv("GOOGLE_API_KEY") or "").strip()
groq_key = (os.getenv("GROQ_API_KEY") or "").strip()

missing = [name for name, val in (("GOOGLE_API_KEY", google_key), ("GROQ_API_KEY", groq_key)) if not val]
if missing:
    st.markdown(
        """<div class="hero"><h1>One quick setup step 🔑</h1>
        <p>The assistant needs its API keys before it can answer questions.</p></div>""",
        unsafe_allow_html=True,
    )
    st.error(f"Missing in your `.env` file: **{', '.join(missing)}**")
    st.markdown("Create a file named **`.env`** in the same folder as `app.py` and add:")
    st.code("GOOGLE_API_KEY=your_google_key_here\nGROQ_API_KEY=your_groq_key_here", language="bash")
    st.markdown("Save the file, then refresh this page.")
    st.stop()


# =====================================================================
# 6) VECTOR DATABASE  (same collection / folder / embedding model as notebook)
# =====================================================================
@st.cache_resource(show_spinner=False)
def get_db(api_key: str):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=api_key,
    )
    return Chroma(
        collection_name="ned_admissions_v1",
        embedding_function=embeddings,
        persist_directory="./ned_admissions_db",
        collection_configuration={"hnsw": {"space": "cosine"}},
    )


def ensure_database(db):
    """First run only: save all policies into the vector database (safe to re-run)."""
    ids = [f"ned_policy_{i:03d}" for i in range(len(policies))]
    existing = set(db.get()["ids"])
    if len(existing) >= len(policies):
        return

    todo = [i for i, pid in enumerate(ids) if pid not in existing]
    batch_size = 10
    with st.status(f"🛠️ First-time setup: saving {len(todo)} NED policies to the database…", expanded=True) as status:
        bar = st.progress(0.0)
        done = 0
        for start in range(0, len(todo), batch_size):
            chunk = todo[start:start + batch_size]
            texts = [policies[i]["text"] for i in chunk]
            metas = [{"category": policies[i]["category"], "topic": policies[i]["topic"]} for i in chunk]
            chunk_ids = [ids[i] for i in chunk]
            for attempt in range(5):
                try:
                    db.add_texts(texts=texts, metadatas=metas, ids=chunk_ids)
                    break
                except Exception as error:
                    if attempt == 4:
                        status.update(label="Setup failed", state="error")
                        raise
                    wait = 15 * (attempt + 1)
                    st.write(f"⏳ API limit reached ({type(error).__name__}). Waiting {wait}s and retrying…")
                    time.sleep(wait)
            done += len(chunk)
            bar.progress(done / len(todo))
            time.sleep(3)
        status.update(label="✅ Database ready! You only see this once.", state="complete", expanded=False)


try:
    ned_db = get_db(google_key)
    ensure_database(ned_db)
except Exception as error:
    st.error(f"Could not prepare the database: **{type(error).__name__}**. Check your GOOGLE_API_KEY and internet connection, then refresh.")
    st.stop()


# =====================================================================
# 7) SEARCH TOOL + LLM + AGENT LOOP  (logic copied exactly from notebook)
# =====================================================================
@tool
def search_ned_admission_rules(query: str) -> str:
    """Search the saved NEDUET admission rules, eligibility criteria, and campus guidelines.
    Use a specific keyword or question targeting the user's inquiry.
    Returns retrieved policy statements with topic metadata, not a generalized answer.
    """
    if not query.strip():
        return json.dumps({"matches": [], "note": "The search query was empty."})

    documents = ned_db.similarity_search(query, k=3)
    evidence = []
    for doc in documents:
        evidence.append({
            "topic": doc.metadata.get("topic", "General"),
            "category": doc.metadata.get("category", "Admissions"),
            "text": doc.page_content
        })
    return json.dumps({"matches": evidence}, ensure_ascii=False)


llm = ChatGroq(
    model=os.getenv("GROQ_CHAT_MODEL", "openai/gpt-oss-20b"),
    temperature=0,
    timeout=60,
    max_retries=1,
)

ned_first_search_llm = llm.bind_tools([search_ned_admission_rules], tool_choice="required")
ned_llm_with_tools = llm.bind_tools([search_ned_admission_rules])


def get_answer_text(response):
    if isinstance(response.content, str):
        return response.content
    if isinstance(response.content, list):
        text_parts = []
        for item in response.content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(item.get("text", ""))
        return "\n".join(text_parts)
    return str(response.content)


def ask_ned_assistant(question: str, show_trace=True):
    question = question.strip()
    if not question:
        return "Please enter a valid question regarding NEDUET Admissions."

    messages = [SystemMessage(content=NED_SYSTEM_RULES), HumanMessage(content=question)]
    search_completed = False

    for turn in range(4):
        model = ned_first_search_llm if turn == 0 else ned_llm_with_tools
        response = model.invoke(messages)

        if response.invalid_tool_calls:
            return "The model produced an invalid query. Please rephrase your question."
        if not response.tool_calls:
            if not search_completed:
                return "Unable to retrieve admission database evidence. Please try again."
            return get_answer_text(response)

        messages.append(response)
        for call in response.tool_calls:
            if call["name"] != "search_ned_admission_rules":
                result = "Requested tool is unavailable."
            else:
                try:
                    result = search_ned_admission_rules.invoke(call["args"])
                    search_completed = True
                    if show_trace:
                        print("[Trace] Query generated by AI:", call["args"].get("query", ""))
                        print("[Trace] Verified database matches:", result)
                except Exception as error:
                    result = f"Search failed due to: {type(error).__name__}. No verified database evidence is available."
            messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

    return "Search-step limit reached. Please ask a simpler question."


# =====================================================================
# 8) UI HELPERS
# =====================================================================
CATEGORY_ICONS = {
    "Admissions & Eligibility": "📝",
    "Entry Test & Syllabus": "🧪",
    "Merit Aggregate & Scoring": "📊",
    "Campus Navigation": "🚌",
    "Fee Structure & Refund Policies": "💳",
    "Examinations & Document Verification": "📄",
    "Campus Procedures & Miscellaneous": "🏫",
    "Faculties & Academic Departments": "🏛️",
    "Postgraduate & Research Programs": "🔬",
    "Historical Background & Milestones": "📜",
    "Student Life & Societies": "🎭",
    "Library & Campus Services": "📚",
    "Accreditation & Professional Bodies": "✅",
    "Examinations & Grading": "🎯",
    "Campus Infrastructure & Operations": "🛠️",
}

QUICK_QUESTIONS = [
    "📝 What is the minimum HSC percentage for BE and BS Computer Science?",
    "📊 How is the merit aggregate calculated?",
    "🧪 What is the entry test format and passing marks?",
    "🩺 I have done Pre-Medical in intermediate. Can I apply for Software Engineering or Bio-Medical?",
    "📄 Which documents do I need to bring for the interview?",
    "🚌 How can I get a transport pass? Is hostel available?",
]


def pretty(text: str) -> str:
    """Display-only touch: make the '[Verified Source: ...]' notes look like green badges."""
    return re.sub(r"\[Verified Source:\s*(.*?)\]", r"✅ :green[*Verified Source: \1*]", text)


# =====================================================================
# 9) SIDEBAR
# =====================================================================
with st.sidebar:
    st.markdown(
        """<div class="side-brand"><div class="logo">🎓</div>
        <div><div class="name">NED Admissions</div><div class="tag">AI Assistant · Since 1922</div></div></div>""",
        unsafe_allow_html=True,
    )
    st.toggle("🌙 Dark theme", key="dark_mode", help="Switch between light and dark look")
    st.button("🗑️ Start a new chat", on_click=clear_chat, use_container_width=True)

    with st.expander("🗂️ Browse topics", expanded=False):
        seen = []
        for p in policies:
            if p["category"] not in seen:
                seen.append(p["category"])
        for i, cat in enumerate(seen):
            st.button(
                f"{CATEGORY_ICONS.get(cat, '📌')} {cat}",
                key=f"cat_{i}",
                on_click=set_pending,
                args=(f"Tell me about {cat} at NED University.",),
                use_container_width=True,
            )

    with st.expander("✍️ Text formatting guide", expanded=False):
        st.markdown(
            """
Answers and your own questions support these styles:

| You type | You see |
|---|---|
| `**important**` | **important** |
| `*note*` | *note* |
| `~~old / not applicable~~` | ~~old / not applicable~~ |
| `***key point***` | ***key point*** |
| `- item` | • item |
| `:green[text]` | :green[text] |
| `:red[text]` | :red[text] |

**Emoji legend**
- ✅ confirmed / verified rule
- ⚠️ be careful, important condition
- 📌 key point to remember
- 📝 documents and forms
- 💡 helpful tip
            """
        )

    st.markdown(
        """<div class="contact-card"><b>📞 Admission Office</b><br>
        +92-21-99261261-8 Ext: 2412<br>
        🌐 <a href="https://www.neduet.edu.pk" target="_blank">www.neduet.edu.pk</a></div>""",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="side-note">Answers come from NED\'s saved admission rules. '
        "For final decisions, always confirm with the official Admission Office.</p>",
        unsafe_allow_html=True,
    )


# =====================================================================
# 10) MAIN CHAT AREA
# =====================================================================
prompt = st.chat_input("Ask anything about NED admission, entry test, fees, campus life…")
if not prompt and st.session_state.pending:
    prompt = st.session_state.pending
st.session_state.pending = None

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

if not st.session_state.messages:
    st.markdown(
        """<div class="hero">
        <div class="est">1922</div>
        <h1>Your NED admission questions, answered.</h1>
        <p>Ask about eligibility, the entry test, merit, fees or campus life in your own words.
        You get a clear answer based on NED's official admission rules.</p>
        <div class="pills"><span class="pill">✅ Verified answers</span><span class="pill">⚡ Instant replies</span><span class="pill">🎓 For future NEDians</span></div>
        </div>""",
        unsafe_allow_html=True,
    )
    with st.chat_message("assistant", avatar="🎓"):
        st.markdown(
            "Assalam o Alaikum! 👋 I'm the **NED Admissions AI Assistant**.\n\n"
            "Type your question below, or pick one from the list. Important points appear in **bold**, "
            "notes in *italics*, and anything outdated as ~~strikethrough~~. 💡"
        )
    st.markdown('<div class="section-title">Popular questions</div>', unsafe_allow_html=True)
    cols = st.columns(2)
    for i, q in enumerate(QUICK_QUESTIONS):
        with cols[i % 2]:
            st.button(q, key=f"quick_{i}", on_click=set_pending, args=(q.split(" ", 1)[1],))

for message in st.session_state.messages:
    avatar = "🧑‍🎓" if message["role"] == "user" else "🎓"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(pretty(message["content"]) if message["role"] == "assistant" else message["content"])

if prompt:
    with st.chat_message("assistant", avatar="🎓"):
        with st.spinner("Searching NED's admission rules…"):
            try:
                answer = ask_ned_assistant(prompt, show_trace=False)
            except Exception as error:
                answer = (
                    "⚠️ **Something went wrong while getting your answer.**\n\n"
                    f"*({type(error).__name__})* Please wait a few seconds and try again. "
                    "If it keeps happening, contact the Admission Office on **+92-21-99261261-8 Ext: 2412**."
                )
        st.markdown(pretty(answer))
    st.session_state.messages.append({"role": "assistant", "content": answer})
