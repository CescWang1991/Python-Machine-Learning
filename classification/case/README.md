# Case: Meeting Supply and Demand

## Introduction
A person makes a doctor appointment, receives all the instructions and no-show. 300k medical appointments and its 15 variables (characteristics) of each. The most important one if the patient show-up or no-show the appointment. Use this 300k medical appointments data to train a model to predicate whether a person will show after he or she makes an appointment.</br>

Dataset available on [kaggle](https://www.kaggle.com/joniarroba/noshowappointments), and [local](../data/KaggleV2-May-2016.csv).

## Data Dictionary
| Feature Name | Description | Type |
| :----- | :----- | :----- |
| PatientId | Identification of a patient | numerical value |
| AppointmentID | Identification of each appointment | numerical value |
| Gender | Male or Female | binary value |
| AppointmentDay | The day of the actual appointment | date |
| ScheduledDay | The day someone called or registered the appointment | date |
| Age | How old is the patient | numerical value |
| Neighbourhood | Where the appointment takes place | categorical value |
| Scholarship | Ture of False | binary value |
| Hipertension | True or False | binary value |
| Diabetes | True or False | binary value |
| Alcoholism | True or False | binary value |
| Handcap | True or False | binary value |
| SMS_received | 1 or more messages sent to the patient | binary value |

| Label Name | Description | Type |
| :----- | :----- | :----- |
| No-show | True or False | categorical value |
