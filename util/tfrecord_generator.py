import os
import sys
import json

# Dependency imports

import tensorflow as tf

def main():
  #qid_dict = {}
  #for line in open('qid.dict'):
  #  flds = line.strip().split('\t')
  #  qid_dict[flds[0]] = long(flds[1])
  filename=sys.argv[1]
  vid_title_dict = {}
  vid_face_dict = {}
  vid_emb_dict = {}
  print "begin to load dict"
  for line in open(sys.argv[2]):
    json_r = json.loads(line.strip())
    vid_title_dict[json_r['item_id']] = [int(x) for x in json_r['title_features'].keys()]
    #print vid_title_dict[json_r['item_id']]
    #break
  print "load title done"
  for line in open(sys.argv[3]):
    json_r = json.loads(line.strip())
    gender = []
    beauty = []
    relative_position = []
    for face in json_r['face_attrs']:
      gender.append(face['gender'])
      beauty.append(int(face['beauty']*100))
      relative_position.append(face['relative_position'])
    #vid_face_dict[json_r['item_id']] = [gender, beauty, relative_position]
    #print vid_face_dict[json_r['item_id']]
    break
  print "load face done"

  for line in open(sys.argv[4]):
    json_r = json.loads(line.strip())
    vid_emb_dict[json_r['item_id']] = json_r['video_feature_dim_128']
    #print vid_emb_dict[json_r['item_id']]
    #break
  print "load emb done"

  n = 0
  writer = tf.python_io.TFRecordWriter("%s_%d.tfrecord" %(filename,n))
  c = 0
  for line in sys.stdin:
    features= line.strip().split('\t')
    c += 1
    uid = int(features[0])
    u_city = int(features[1])
    item_id = int(features[2])
    author_uid = int(features[3])
    i_city = int(features[4])
    channel = int(features[5])
    finish = int(features[6])
    like = int(features[7])
    music_id = int(features[8])
    device_id = int(features[9])
    create_time = int(features[10])
    item_duration = int(features[11])
    titles = vid_title_dict[item_id] if item_id in vid_title_dict else []
    gender,beauty,relative_position = vid_face_dict[item_id] if item_id in vid_face_dict else [],[],[]
    i_emb = vid_emb_dict[item_id] if item_id in vid_emb_dict else []
    example = tf.train.Example(features=tf.train.Features(feature={
    'uid': tf.train.Feature(int64_list=tf.train.Int64List(value=[uid]))
    ,'u_city': tf.train.Feature(int64_list=tf.train.Int64List(value=[u_city]))
    ,'item_id': tf.train.Feature(int64_list=tf.train.Int64List(value=[item_id]))
    ,'author_uid': tf.train.Feature(int64_list=tf.train.Int64List(value=[author_uid]))
    ,'i_city': tf.train.Feature(int64_list=tf.train.Int64List(value=[i_city]))
    ,'channel': tf.train.Feature(int64_list=tf.train.Int64List(value=[channel]))
    ,'finish': tf.train.Feature(int64_list=tf.train.Int64List(value=[finish]))
    ,'like': tf.train.Feature(int64_list=tf.train.Int64List(value=[like]))
    ,'music_id': tf.train.Feature(int64_list=tf.train.Int64List(value=[music_id]))
    ,'device_id': tf.train.Feature(int64_list=tf.train.Int64List(value=[device_id]))
    ,'item_duration': tf.train.Feature(int64_list=tf.train.Int64List(value=[item_duration]))
    ,'title': tf.train.Feature(int64_list=tf.train.Int64List(value=titles[0:20]))
    ,'gender': tf.train.Feature(int64_list=tf.train.Int64List(value=gender))
    ,'beauty': tf.train.Feature(int64_list=tf.train.Int64List(value=beauty))
    ,'item_emb': tf.train.Feature(float_list=tf.train.FloatList(value=i_emb))
    #,'cover_feature': tf.train.Feature(float_list=tf.train.FloatList(value=values[15:415]))
    }))
    if (c % 10000) == 0:
      print >> sys.stderr, "%s done" %(c)
    serialized = example.SerializeToString()
    writer.write(serialized)
    if c % 1000000 == 0:
      writer.close()
      n += 1
      writer = tf.python_io.TFRecordWriter("%s_%d.tfrecord" %(filename,n))
  writer.close()


main()
