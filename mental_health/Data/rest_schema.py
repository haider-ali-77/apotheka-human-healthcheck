output_response = {
        'plots':[#a list of dictionary of plots
            
        {
        'plot_type':'histogram',#str:str
        'image':'png_serialized_image_string_dtype=uint8',#str:str
        'image_height':1280,
        'image_width': 720,
        'image_channels':3,
        'plot_title':'title of the plot',
        'plot_description':'a small helpful description',#str:str
        'target_feature':'feature_name'#str:str
        },# and so on

    ],
    'descriptive_stats':[# a list of desriptive stats and features in the stat
       ['stat1','feature_name1','feature_name2'],#and so on
         
        ]
    }
    
input_dict = {
    'dataset':[# list of list of values of the 471 features. Shape is Nx471
        ['feature1_value','feature2_value','feature3_value',...., 'feature471_value'],# and so on...
        
        ],
    'plot_requirements':{# an optional dictionary of requiremnets for output plots
        'image_width':720,#str:Union[int,None]
        'image_height':1280#str:Union[int,None]
        }
    }
        
        
    
