def get_bball_split(dataset):
    seqs = [
         'player_event_0',
         'player_event_1',
         'player_event_2',
         'player_event_3',
         'player_event_4',
         'player_event_5',
         'player_event_6',
         'player_event_7',
         'player_event_8',
         'player_event_9',
         'player_event_10',
         'player_event_11',
         'player_event_12',
         'player_event_13',
         'player_event_14',
         'player_event_15',
         'player_event_16',
         'player_event_17',
         'player_event_18',
         'player_event_20',
         'player_event_21',
         'player_event_22',
         'player_event_24',
         'player_event_25',
         'player_event_26',
         'player_event_27',
         'player_event_28',
         'player_event_29',
         'player_event_30',
         'player_event_31',
         'player_event_32',
         'player_event_33',
         'player_event_34',
         'player_event_35',
         'player_event_36',
         'player_event_37',
         'player_event_38',
         'player_event_39',
         'player_event_40',
         'player_event_41',
         'player_event_42',
         'player_event_43',
         'player_event_44'
]
     
    if dataset == 'basketball':
        test = ['player_event_0_test',
               'player_event_1_test',
               'player_event_2_test',
               'player_event_3_test',
               'player_event_4_test',
               'player_event_5_test',
               'player_event_6_test',
               'player_event_7_test',
               'player_event_8_test',
               'player_event_9_test',
               'player_event_10_test',
               'player_event_11_test',
               'player_event_12_test',
               'player_event_13_test',
               'player_event_14_test',
               'player_event_15_test',
               'player_event_16_test',
               'player_event_17_test',
               'player_event_18_test',
               'player_event_19_test',
               'player_event_20_test',
               'player_event_21_test',
               'player_event_22_test',
               'player_event_23_test',
               'player_event_24_test',
               'player_event_25_test',
               'player_event_26_test',
               'player_event_27_test',
               'player_event_28_test',
               'player_event_29_test',
               'player_event_30_test',
               'player_event_31_test',
               'player_event_32_test',
               'player_event_33_test',
               'player_event_34_test',
               'player_event_35_test',
               'player_event_36_test',
               'player_event_37_test',
               'player_event_38_test',
               'player_event_39_test',
               'player_event_40_test',
               'player_event_41_test',
               'player_event_42_test',
               'player_event_43_test',
               'player_event_44_test',
               'player_event_45_test',
               'player_event_46_test',
               'player_event_47_test',
               'player_event_48_test',
               'player_event_49_test',
               'player_event_50_test',
               'player_event_51_test',
               'player_event_52_test',
               'player_event_53_test',
               'player_event_54_test',
               'player_event_55_test',
               'player_event_56_test',
               'player_event_57_test',
               'player_event_58_test',
               'player_event_59_test',
               'player_event_60_test',
               'player_event_61_test',
               'player_event_62_test',
               'player_event_63_test']

    train, val = [], []
    for seq in seqs:
        train.append(f'{seq}_train')
        val.append(f'{seq}_val')
    return train, val, test