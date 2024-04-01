import json

# Define the dictionary
data = {
    "TOP22_006_CH_LST_SR": {
        "2l": {
            "lep_chan_lst": ["2lss_p", "2lss_m", "2lss_4t_p", "2lss_4t_m"],
            "lep_flav_lst": ["ee", "em", "mm"],
            "appl_lst": ["isSR_2lSS", "isAR_2lSS"],
            "appl_lst_data": ["isAR_2lSS_OS"],
            "jet_lst": ["exactly_4j", "exactly_5j", "exactly_6j", "atleast_7j"]
        },
        "3l": {
            "lep_chan_lst": ["3l_p_offZ_1b", "3l_m_offZ_1b", "3l_p_offZ_2b", "3l_m_offZ_2b", "3l_onZ_1b", "3l_onZ_2b"],
            "lep_flav_lst": ["eee", "eem", "emm", "mmm"],
            "appl_lst": ["isSR_3l", "isAR_3l"],
            "jet_lst": ["exactly_2j", "exactly_3j", "exactly_4j", "atleast_5j"]
        },
        "4l": {
            "lep_chan_lst": ["4l"],
            "lep_flav_lst": ["llll"],
            "appl_lst": ["isSR_4l"],
            "jet_lst": ["exactly_2j", "exactly_3j", "atleast_4j"]
        }
    },

    "OFFZ_SPLIT_CH_LST_SR": {
        "2l": {
            "lep_chan_lst": ["2lss_p", "2lss_m", "2lss_4t_p", "2lss_4t_m"],
            "lep_flav_lst": ["ee", "em", "mm"],
            "appl_lst": ["isSR_2lSS", "isAR_2lSS"],
            "appl_lst_data": ["isAR_2lSS_OS"],
            "jet_lst": ["exactly_4j", "exactly_5j", "exactly_6j", "atleast_7j"]
        },
        "3l": {
            "lep_chan_lst": [
                "3l_p_offZ_low_1b", "3l_p_offZ_high_1b", "3l_p_offZ_none_1b",
                "3l_m_offZ_low_1b", "3l_m_offZ_high_1b", "3l_m_offZ_none_1b",
                "3l_p_offZ_low_2b", "3l_p_offZ_high_2b", "3l_p_offZ_none_2b",
                "3l_m_offZ_low_2b", "3l_m_offZ_high_2b", "3l_m_offZ_none_2b",
                "3l_onZ_1b", "3l_onZ_2b"
            ],
            "lep_flav_lst": ["eee", "eem", "emm", "mmm"],
            "appl_lst": ["isSR_3l", "isAR_3l"],
            "jet_lst": ["exactly_2j", "exactly_3j", "exactly_4j", "atleast_5j"]
        },
        "4l": {
            "lep_chan_lst": ["4l"],
            "lep_flav_lst": ["llll"],
            "appl_lst": ["isSR_4l"],
            "jet_lst": ["exactly_2j", "exactly_3j", "atleast_4j"]
        }
    }

    "CH_LST_CR": {
        "2l_CRflip" : {
            "lep_chan_lst": ["2lss_CRflip"],
            "lep_flav_lst": ["ee"],
            "appl_lst": ["isSR_2lSS" , "isAR_2lSS"],
            "appl_lst_data": ["isAR_2lSS_OS"],
            "jet_lst": ["atmost_3j"]
        },
        "2l_CR" : {
            "lep_chan_lst": ["2lss_CR"],
            "lep_flav_lst": ["ee" , "em" , "mm"],
            "appl_lst": ["isSR_2lSS" , "isAR_2lSS"],
            "appl_lst_data": ["isAR_2lSS_OS"],
            "jet_lst": ["exactly_1j", "exactly_2j", "exactly_3j"]
        },
        "3l_CR" : {
            "lep_chan_lst": ["3l_CR"],
            "lep_flav_lst": ["eee" , "eem" , "emm", "mmm"],
            "appl_lst": ["isSR_3l" , "isAR_3l"],
            "jet_lst": ["exactly_0j", "atleast_1j"]
        },
        "2los_CRtt" : {
            "lep_chan_lst": ["2los_CRtt"],
            "lep_flav_lst": ["em"],
            "appl_lst": ["isSR_2lOS" , "isAR_2lOS"],
            "jet_lst": ["exactly_2j"]
        },
        "2los_CRZ" : {
            "lep_chan_lst": ["2los_CRZ"],
            "lep_flav_lst": ["ee", "mm"],
            "appl_lst": ["isSR_2lOS" , "isAR_2lOS"],
            "jet_lst": ["atleast_0j"]
        }
    }
}

# Serialize the dictionary into a JSON file
with open('data.json', 'w') as json_file:
    json.dump(data, json_file, indent=4)

print("Serialized dictionary to data.json")

# Deserialize the JSON from the file
with open('data.json', 'r') as json_file:
    deserialized_data = json.load(json_file)

print("\nDeserialized dictionary from data.json:")
print(deserialized_data)
