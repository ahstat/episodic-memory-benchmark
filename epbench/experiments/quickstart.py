# Default file paths to the data folder and the environment variable
from pathlib import Path
git_repo_filepath = '/filepath/to/gitrepo/episodic-memory-benchmark'
data_folder = Path(git_repo_filepath) / 'epbench' / 'data'
env_file = Path(git_repo_filepath) / '.env'

# Parsing the arguments
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data_folder', type=str, default=str(data_folder),
                    help='Path to the data folder')
parser.add_argument('--env_file', type=str, default=str(env_file),
                    help='Path to the .env file')
parser.add_argument('--book_nb_events', type=int, default=20,
                    help='Number of events in the book (20 for short and 200 for long, for the default experiment)')
parser.add_argument('--answering_kind', type=str, default='prompting',
                    help='Answering kind')
parser.add_argument('--answering_model_name', type=str, default='gpt-4o-mini-2024-07-18',
                    help='Answering model name')

# Overrid the file paths
args = parser.parse_args()
data_folder = Path(args.data_folder)
env_file = Path(args.env_file)

# Step 1: generating the synthetic episodic memory dataset

## Configuration (here, default short book with 20 events)
book_parameters = {
  'indexing': 'default', 
  'nb_summaries': 0
  }
prompt_parameters = {
  'nb_events': args.book_nb_events, 
  'name_universe': 'default', 
  'name_styles': 'default', 
  'seed': 0, 
  'distribution_events': {
    'name': 'geometric', 
    'param': 0.1
    }
  }
model_parameters = {
  'model_name': 'claude-3-5-sonnet-20240620', 
  'max_new_tokens': 4096, 
  'itermax': 10
  }

## Generation (generate the book, then compute the ground truth QAs)
from epbench.src.generation.benchmark_generation_wrapper import BenchmarkGenerationWrapper
my_benchmark = BenchmarkGenerationWrapper(
  prompt_parameters, model_parameters, book_parameters, data_folder, env_file)

# Step 2: predicting the answers given the document and the questions

## Configuration
answering_parameters = {
  'kind': args.answering_kind, 
  'model_name': args.answering_model_name, 
  'max_new_tokens': 4096, 
  'sleeping_time': 0, 
  'policy': 'remove_duplicates'
  }

## Prediction (generate answers, then evaluate them)
from epbench.src.evaluation.evaluation_wrapper import EvaluationWrapper
my_evaluation = EvaluationWrapper(my_benchmark, answering_parameters, data_folder, env_file)

# Step 3: extract the performance results

## Configuration
experiments = [{
  'book_nb_events': args.book_nb_events, 
  'book_model_name': 'claude-3-5-sonnet-20240620',
  'answering_kind': args.answering_kind, 
  'answering_model_name': args.answering_model_name,
  'answering_embedding_chunk': 'n/a'
  },
]
all_benchmarks = {f'benchmark_claude_default_{args.book_nb_events}': my_benchmark}

## Results
from epbench.src.evaluation.precomputed_results import get_precomputed_results
df = get_precomputed_results(experiments, env_file, data_folder, all_benchmarks)
df

from epbench.src.results.average_groups import extract_groups
# select the book of interest (either 20 or 200)
nb_events = args.book_nb_events
# select the elements to group
relative_to = ['get', 'bins_items_correct_answer']
# group the results according to `relative_to`
df_results = extract_groups(df, nb_events, relative_to)
# further filtering by selecting only the simple recall questions
df_results = df_results[df_results['get'] == 'all'].drop('get', axis = 1)

print(df_results)
print('Ended successfully')
