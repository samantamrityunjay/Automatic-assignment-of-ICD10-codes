# Script to extract different sections from discharge summary
# This script only extracts family and social history
# Coordinate with Akshay because this script is common between both pipelines


import re

# Section titles present in discharge summaries


def family_social_sections(text):
    para = re.split("\n", text)
    social = ""
    family = ""
    section_indices = []
    for p in para:
        if re.search(r"^([A-Z](.*?):)", p):
            section_indices.append(para.index(p))
    for i in range(len(section_indices) - 1):
        header = re.findall(r"^([A-Z](.*?):)", para[section_indices[i]])[0][0].replace("\n", "").replace(":",
                                                                                                         "").lower()
        if header in social_family_history_headers['social']:
            replaced = re.sub(r"^([A-Z](.*?):)", "", para[section_indices[i]])
            social += " " + replaced + " ".join(para[section_indices[i] + 1: section_indices[i + 1]]) + "\n"
        elif header in social_family_history_headers['family']:
            replaced = re.sub(r"^([A-Z](.*?):)", "", para[section_indices[i]])
            family += " " + replaced + " ".join(para[section_indices[i] + 1: section_indices[i + 1]]) + "\n"
    family = family.replace("\n", " ").strip()
    social = social.replace("\n", " ").strip()
    return family, social


def diagnosis_med_hist_sections(text):
    sections = {}
    for key in med_history_diagnosis_section_headers.keys():
        search_list = med_history_diagnosis_section_headers[key]
        extracted_text = {}
        for search_word in search_list:
            # extracted_text[i] = ''
            flag = 'True'
            temp_text = ''
            text2 = text
            if search_word in med_history_diagnosis_section_headers[key]:
                for n, m in extracted_text.items():
                    if search_word.lower() + ':' in m.lower():
                        temp = re.search(search_word.lower() + ":", m.lower())
                        extracted_text[n] = m[:temp.start()]
                        extracted_text[search_word] = m[temp.end() + 1:]
                        flag = 'False'
                        break
            try:
                if flag == 'True':
                    temp = re.search("\n" + search_word.lower() + ":", text2.lower())
                    text2 = text2[temp.end() + 1:]
                    temp2 = re.search("(\n\n.*:)", text2)
                    temp_text = temp_text + '\n' + text2[:temp2.start()]
                    if temp_text != '':
                        extracted_text[search_word] = temp_text
            except:
                pass
        sections[key] = extracted_text
    return sections


social_family_history_headers = {'social': ['social history',
                              'social hx'],
                   'family': ['family history',
                              'family hx'],
                                 }
med_history_diagnosis_section_headers = {
    'diagnosis':
        [
            'discharge diagnoses',
            'discharge diagnosis',
            'final diagnoses',
            'Seconday diagnoses',
            'Seconday diagnosis',
            'final diagnosis',
            'primary diagnoses',
            'primary diagnosis',
            'secondary diagnoses',
            'secondary diagnosis'
        ],
    'medical_history':
        [
            'Past Medical History',
            'history of present illness',
            'history of the present illness'
        ]
}