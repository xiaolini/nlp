#encoding:utf-8
import os

#在命名体识别任务（NER，Named Entity Recognizer）中,Evaluate使用Perl的conlleval.pl
def conlleval(label_predict, label_path, metric_path):
    """
    :param label_predict:
    :param label_path:
    :param metric_path:
    :return:
    """
    eval_perl = "./conlleval_rev.pl"
    with open(label_path, "w") as fw:
        line = []
        for sent_result in label_predict:
            for char, tag, tag_ in sent_result:
                tag = '0' if tag == 'O' else tag
                char = char.encode("utf-8")
                line.append("{} {} {}\n".format(char, tag, tag_))
            line.append("\n")
        fw.writelines(line)
    os.system("perl {} < {} > {}".format(eval_perl, label_path, metric_path))
    with open(metric_path) as fr:
        metrics = [line.strip() for line in fr]
    return metrics