# coding: utf-8

# Usage
# python src/run_parallel.py src/tasks_config.challenge.json --worlds 128 --reduction-interval 100 --episodes 10

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os
import logging
import logging.config
import operator
from optparse import OptionParser
from core.serializer import StandardSerializer
from core.environment import Environment
from core.config_loader import JSONConfigLoader, PythonConfigLoader
import learners
from core.session import Session

from multiprocessing import Process, Pipe
import numpy as np
import pickle

def arg_parse():
    op = OptionParser("Usage: %prog [options] "
                      "(tasks_config.json | tasks_config.py)")
    op.add_option('-o', '--output', default='result',
                  help='File where the simulation results(model weight) are saved.')
    op.add_option('--scramble', action='store_true', default=False,
                  help='Randomly scramble the words in the tasks for '
                  'a human player.')
    op.add_option('-w', '--show-world', action='store_true', default=False,
                  help='shows a visualization of the world in the console '
                  '(mainly for debugging)')
    op.add_option('-d', '--time-delay', default=0, type=float,
                  help='adds some delay between each timestep for easier'
                  ' visualization.')
    op.add_option('-l', '--learner',
                  default='learners.es_learner.ESLearner',
                  help='Defines the type of learner.')
    op.add_option('-v', '--view',
                  default='BaseView',
                  help='Viewing mode.')
    op.add_option('--learner-cmd',
                  help='The cmd to run to launch RemoteLearner.')
    op.add_option('--learner-port',
                  default=5556,
                  help='Port on which to accept remote learner.')
    op.add_option('--max-reward-per-task',
                  default=2147483647, type=int,
                  help='Maximum reward that we can give to a learner for'
                  ' a given task.')
    op.add_option('--curses', action='store_true', default=False,
                  help='Uses standard output instead of curses library.')
    op.add_option('--bit-mode', action='store_true', default=False,
                  help='Environment receives input in bytes.')
    op.add_option('--worlds', default=10, type=int,
                  help='The number of process used for searching.')
    op.add_option('--reduction-interval', default=500, type=int,
                  help='weights are updated every reduction interval')
    op.add_option('--episodes', default=100, type=int,
                  help='the number of episodes')
    opt, args = op.parse_args()
    if len(args) == 0:
        op.error("Tasks schedule configuration file required.")
    tasks_config_file = args[0]
    return args, opt

def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    args, opt = arg_parse()
    tasks_config_file = args[0]
    
    # MAKE SEED_WEIGHT
    logger.info("Sampling seed weight")
    # TODO: read weight from pickle file pointed if exists
    seed_weight = None
    master_learner = create_learner(opt.learner, None, opt.learner_cmd, opt.learner_port, not opt.bit_mode)
    master_learner.net.set_weight(seed_weight)
    
    # MAKE PROCESSES
    logger.info("Starting new processes")
    process_info = []
    for i in range(opt.worlds):    
        p_conn, c_conn = Pipe()
        p = Process(target=process_world, args=(c_conn, opt, tasks_config_file, i))
        p.daemon = True
        p.start()
        process_info.append((p, p_conn))
    
    # LEARNING LOOP & REDUCTION
    logger.info("learning loops begin")
    reward_list = None
    for episode_id in range(opt.episodes):
        weight = master_learner.net.get_weight()
        for p, p_conn in process_info:
            p_conn.send((episode_id, opt.reduction_interval, np.random.randint(0, np.iinfo(np.int32).max), weight))
        reward_list = []
        for p, p_conn in process_info:
            ret = p_conn.recv()
            if ret is None:
                raise Exception('sub process error')
            reward_list.append(ret)
        reward_mean = float(np.asarray([t[0] for t in reward_list]).mean())
        master_learner.net.move_base_weight(reward_list)
        with open(opt.output + '_weight.pklb', 'wb') as f:
            pickle.Pickler(f, protocol=2).dump(master_learner.net.get_weight())
        logger.info('episode %d %f\n%s'%(episode_id, reward_mean, str(reward_list)) )
        with open(opt.output + '_reward.txt', 'a') as f:
            print('episode %d %f\n%s'%(episode_id, reward_mean, str(reward_list)), file=f)
        
    for p, p_conn in process_info:
        p_conn.send(None)
    
    logger.info("all ended")

def process_world(conn, opt, tasks_config_file, world_id):
    try:
        serializer = StandardSerializer()
        task_scheduler = create_tasks_from_config(tasks_config_file)
        env = Environment(serializer, task_scheduler, opt.scramble, opt.max_reward_per_task, not opt.bit_mode)
        learner = create_learner(opt.learner, serializer, opt.learner_cmd, opt.learner_port, not opt.bit_mode)
        session = Session(env, learner, opt.time_delay)
        
        args = conn.recv()
        while not (args is None):
            episode_id, step_count, seed, weight = args
            # INTERACTION BETWEEN ENVIRONMENT AND AGENT
            learner.net.set_genotype_weight(weight, seed)
            del weight
            episode_reward = session.iterate_n(step_count)
            # save_results(session, opt.output)  
            conn.send((episode_reward, seed))
            args = conn.recv()
    except BaseException as e:
        print(e)
        conn.send(None)
    
def create_view(view_type, learner_type, env, session, serializer, show_world, use_curses, byte_mode):
    if not use_curses:
        from view.win_console import StdInOutView, StdOutView
        if learner_type.split('.')[0:2] == ['learners', 'human_learner'] \
           or view_type == 'ConsoleView':
            return StdInOutView(env, session, serializer, show_world, byte_mode)
        else:
            return StdOutView(env, session)
    else:
        from view.console import ConsoleView, BaseView
        if learner_type.split('.')[0:2] == ['learners', 'human_learner'] \
                or view_type == 'ConsoleView':
            return ConsoleView(env, session, serializer, show_world, byte_mode)
        else:
            return BaseView(env, session)

def create_learner(learner_type, serializer, learner_cmd, learner_port=None, byte_mode=False):
    if learner_type.split('.')[0:2] == ['learners', 'human_learner']:
        c = learner_type.split('.')[2]
        if c == 'HumanLearner':
            return learners.human_learner.HumanLearner(serializer, byte_mode)
        elif c == 'ImmediateHumanLearner':
            return learners.human_learner.ImmediateHumanLearner(serializer, byte_mode)
        elif c == 'HaltOnDotHumanLearner':
            return learners.human_learner.HaltOnDotHumanLearner(serializer, byte_mode)
    else:
        # dynamically load the class given by learner_type
        # separate the module from the class name
        path = learner_type.split('.')
        mod, cname = '.'.join(path[:-1]), path[-1]
        # import the module (and the class within it)
        m = __import__(mod, fromlist=[cname])
        c = getattr(m, cname)
        # instantiate the learner

        return c(learner_cmd, learner_port) if 'RemoteLearner' in cname else c()

def create_tasks_from_config(tasks_config_file):
    ''' Returns a TaskScheduler based on either:

        - a json configuration file.
        - a python module with a function create_tasks that does the job
        of returning the task scheduler.
    '''
    fformat = tasks_config_file.split('.')[-1]
    if fformat == 'json':
        config_loader = JSONConfigLoader()
    elif fformat == 'py':
        config_loader = PythonConfigLoader()
    else:
        raise RuntimeError("Unknown configuration file format '.{fformat}' of"
                           " {filename}"
                           .format(fformat=fformat,
                                   filename=tasks_config_file))
    return config_loader.create_tasks(tasks_config_file)

def save_results(session, output_file):
    if session.get_total_time() == 0:
        # nothing to save
        return
    with open(output_file, 'w') as fout:
        print('* General results', file=fout)
        print('Average reward: {avg_reward}'.format(
            avg_reward=session.get_total_reward() / session.get_total_time()),
            file=fout)
        print('Total time: {time}'.format(time=session.get_total_time()),
               file=fout)
        print('Total reward: {reward}'.format(
            reward=session.get_total_reward()),
            file=fout)
        print('* Average reward per task', file=fout)
        for task, t in sorted(session.get_task_time().items(),
                              key=operator.itemgetter(1)):
            r = session.get_reward_per_task()[task]
            print('{task_name}: {avg_reward}'.format(
                task_name=task, avg_reward=r / t),
                file=fout)
        print('* Total reward per task', file=fout)
        for task, r in sorted(session.get_reward_per_task().items(),
                              key=operator.itemgetter(1), reverse=True):
            print('{task_name}: {reward}'.format(task_name=task, reward=r),
                  file=fout)
        print('* Total time per task', file=fout)
        for task, t in sorted(session.get_task_time().items(),
                              key=operator.itemgetter(1)):
            print('{task_name}: {time}'.format(task_name=task, time=t),
                  file=fout)
        print('* Number of trials per task', file=fout)
        for task, r in sorted(session.get_task_count().items(),
                              key=operator.itemgetter(1)):
            print('{task_name}: {reward}'.format(task_name=task, reward=r),
                  file=fout)

def setup_logging(
    default_path='logging.ini',
    default_level=logging.INFO,
    env_key='LOG_CFG'
):
    """Setup logging configuration

    """
    path = default_path
    value = os.getenv(env_key, None)
    if value:
        path = value
    if os.path.exists(path):
        logging.config.fileConfig(default_path)
    else:
        logging.basicConfig(level=default_level)

if __name__ == '__main__':
    main()
