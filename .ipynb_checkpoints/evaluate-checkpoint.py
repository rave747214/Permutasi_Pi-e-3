import os
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')

def generate_top_csv(data='A', top=9, metrics=['loss', 'f1'], save=True):
    ls = sorted(os.listdir('csv/'))
    ls = [x for x in ls if 'metrics.%s' % data in x]
    df = pd.concat([pd.read_csv('csv/' + x) for x in ls])
    
    df['data.bs.nhid'] = ls
    df['data.bs.nhid'] = df['data.bs.nhid'].apply(lambda x : x[8:-4])
    df = df.set_index('data.bs.nhid')
    
    select = []
    for x in metrics:
        select.extend(['train_%s' % x, 'val_%s' % x, 'test_%s' % x])
    df = df[select]
    df = df.sort_values(by='test_f1', ascending=False)
    df = df.iloc[:top]
    if save:
        df.to_csv('best_model_%s_by_f1.csv' % data)
    return df
        
def generate_top9_chart(data):
    top = generate_top_csv(data, 9, save=False)
    histories = [[pd.read_csv('csv/hist.%s.csv' % top.index[3*i + j]) \
                  for j in range(3)] for i in range(3)]

    figsize = (12, 12)
    fig, ax = plt.subplots(3, 3)
    fig.set_size_inches(figsize)

    for i in range(3):
        for j in range(3):
            history = histories[i][j]
            history = history[:100]
            ax[i][j].plot(history['loss'])
            ax[i][j].plot(history['val_loss'])
            ax[i][j].plot(history['f1'])
            ax[i][j].plot(history['val_f1'])
            ax[i][j].set_title('Model ' + top.index[3*i + j])

    ax[2][1].set_xlabel('epoch')
    plt.suptitle('Performa Pelatihan\nPada Dataset %s' % data, \
                 fontsize=15)
    fig.legend(['train_loss', 'val_loss', 'train_f1', 'val_f1'], \
               loc='upper left', fontsize=15)
    plt.savefig('figures/best_9_%s.png' % data)
    
def main():
    generate_top_csv('A', metrics=['f1'])
    generate_top_csv('B', metrics=['f1'])
    generate_top9_chart('A')
    generate_top9_chart('B')
    
if __name__ == '__main__':
    main()