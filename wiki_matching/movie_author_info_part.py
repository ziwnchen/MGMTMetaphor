import pandas as pd
import numpy as np
import time
from tqdm import tqdm
tqdm.pandas()
from wikidata.client import Client

def property_value_extraction(entity, prop):
    claims = entity.data.get('claims', {})
    property_claims = claims.get(prop, []) # usually start with p
    values = []
    for claim in property_claims:
        mainsnak = claim.get('mainsnak', {})
        datavalue = mainsnak.get('datavalue', {})
        value = datavalue.get('value', {}).get('id', None)
        if value:
            values.append(value)
    return values

def time_extraction(entity, prop):
    claims = entity.data.get('claims', {})
    property_claims = claims.get(prop, []) # usually start with p
    values = []
    for claim in property_claims:
        mainsnak = claim.get('mainsnak', {})
        datavalue = mainsnak.get('datavalue', {})
        value = datavalue.get('value', {}).get('time', None)
        if value:
            values.append(value)
    return values

# sometimes, education information is nested
def get_education(entity):
    # if degree or major is nested in education
    institution_values = []
    degree_values = []
    major_values = []

    claims = entity.data.get('claims', {})
    education_claims = claims.get('P69', []) # usually start with p
    for education in education_claims:
        # main entry: institution
        mainsnak = education.get('mainsnak', {})
        datavalue = mainsnak.get('datavalue', {})
        value = datavalue.get('value', {}).get('id', None)
        if value:
            institution_values.append(value)
        # search qualifiers for degree and major, in case they are nested
        qualifiers = education.get('qualifiers', {})
        if "P512" in qualifiers:
            degree_id = qualifiers["P512"][0].get('datavalue', {}).get('value', {}).get('id')
            degree_values.append(degree_id)
        if "P812" in qualifiers:
            major_id = qualifiers["P812"][0].get('datavalue', {}).get('value', {}).get('id')
            major_values.append(major_id)
    
    # search for degree and major if they are not nested
    degree_values += property_value_extraction(entity, 'P512')
    major_values += property_value_extraction(entity, 'P812')

    # remove duplicates
    institution_values = list(set(institution_values))
    degree_values = list(set(degree_values))
    major_values = list(set(major_values))

    return institution_values, degree_values, major_values    

def get_SES_characteristics(entity):
    # gender
    gender_values = property_value_extraction(entity, 'P21')
    # birth date
    birth_date_values = time_extraction(entity, 'P569')
    # death date
    death_date_values = time_extraction(entity, 'P570')
    # citizenship
    citizenship_values = property_value_extraction(entity, 'P27')
    # ethinic group
    ethinic_group_values = property_value_extraction(entity, 'P172')
    # education
    education_values, degree_values, major_values = get_education(entity)
    # student of (this will be sparse) -- if students of famous economists are included, this will be useful
    student_of_values = property_value_extraction(entity, 'P1066')
    # occupation
    occupation_values = property_value_extraction(entity, 'P106')
    # employer in the past
    employer_values = property_value_extraction(entity, 'P108')
    # political party (this will be sparse)
    political_party_values = property_value_extraction(entity, 'P102')
    # ideology (this will be sparse)
    ideology_values = property_value_extraction(entity, 'P1142')

    # dictionary
    SES_characteristics = {"gender": gender_values, "birth_date": birth_date_values, "death_date": death_date_values,
                            "citizenship": citizenship_values, "ethinic_group": ethinic_group_values, "education": education_values,
                            "degree": degree_values,"major": major_values, "student_of": student_of_values, "occupation": occupation_values,
                            "employer": employer_values, "political_party": political_party_values, "ideology": ideology_values}
    return SES_characteristics

# total entities to be queried
data_path = "/zfs/projects/faculty/amirgo-management/opus/processed/"
df_matches = pd.read_pickle(data_path+"imdb_wikidata_person_mapping.pkl")
print("Number of unique congress member: ", len(df_matches["wikidata_id"].unique()))
total_qids = df_matches["wikidata_id"].unique()

# for each entity, get SES characteristics
client = Client()
total_ses_characteristics = []
error_qids = []
for qid in tqdm(total_qids):
    try:
        entity = client.get(qid, load=True)
        SES_characteristics = get_SES_characteristics(entity)
        SES_characteristics["qid"] = qid
        total_ses_characteristics.append(SES_characteristics)
        # wait for 0.2 second
        time.sleep(0.2)
    except:
        print("Error: ", qid)
        error_qids.append(qid)
        time.sleep(30) # wait for 30 seconds

# save SES characteristics
ses_df = pd.DataFrame(total_ses_characteristics)
ses_df.to_pickle(data_path + "opus_ses_characteristics.pkl")
if len(error_qids) > 0:
    pd.Series(error_qids).to_pickle(data_path + "opus_ses_characteristics_error.pkl")