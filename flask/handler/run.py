import json

import train

if __name__ == '__main__':
    import json
 
    with open('config.json', 'r') as f:
        opt = json.load(f)
    
    trainer = train.Trainer(opt)
    trainer.fit()