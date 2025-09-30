# Events, meta-events, and prompting
from epbench.src.generation.generate_1_events_and_meta_events import generate_and_export_events_and_meta_events_func
# Generate (or load existing) events
from epbench.src.generation.generate_2_paragraph import iterative_generate_paragraphs_func
# Post-processing (catch invalid samples, remove the numbering, replace entities)
from epbench.src.generation.generate_3_secondary_entities import get_final_samples, count_tokens
# Build book
from epbench.src.generation.generate_4_books import book_indexing_func, build_chapter
# Build questions
from epbench.src.generation.qa_generation import build_qa_func, checking_widespreadness_of_questions
# IO
import pandas as pd
import pyarrow
import ast
from epbench.src.io.io import book_dirpath_func, export_list, import_list
# Printing
import numpy as np
from epbench.src.generation.printing import pretty_print_before_post_processing, pretty_print, wrap_text, split_chapters_func, find_relative_position
# Fine-tuning
from random import shuffle
from epbench.src.generation.qa_generation import retrieve_dataframe_for_single_question, build_nonempty_qa_func
# For reference
from epbench.src.generation.raw_materials import parameters_universe_func
from epbench.src.generation.generate_1_events_and_meta_events import generate_universe
# Plotting
from pathlib import Path
from epbench.src.plots.plotting_functions import plotting_ecdf

class BenchmarkGenerationWrapper:
    def __init__(
            self,
            prompt_parameters = {
                'nb_events': 20, 
                'name_universe': 'default', 
                'name_styles': 'default', 
                'seed': 0, 
                'distribution_events': {'name': 'geometric', 'param': 0.1}
            },
            model_parameters = {
                'model_name': 'claude-3-5-sonnet-20240620',
                'max_new_tokens': 4096, 
                'itermax': 10 # itermax is integer, 1 for a single try
            },
            book_parameters = {
                'indexing': 'default'
            },
            data_folder = '/repo/to/git/main/epbench/data',
            env_file = '/repo/to/git/main/.env',
            verbose = True,
            rechecking = True
            ):
        book, df_book_groundtruth, df_qa, df_qa_debug_widespreadness, debug_all_generated_samples = self.__end2end(
            prompt_parameters, model_parameters, book_parameters, data_folder, env_file, verbose, rechecking)
        self.book = book
        self.df_book_groundtruth = df_book_groundtruth
        self.df_qa = df_qa
        self.df_qa_debug_widespreadness = df_qa_debug_widespreadness
        self.debug_all_generated_samples = debug_all_generated_samples

        # input values
        self.prompt_parameters = prompt_parameters
        self.model_parameters =  model_parameters
        self.book_parameters = book_parameters
        self.data_folder = data_folder
        self.env_file = env_file
        self.verbose = verbose
        self.rechecking = rechecking

        # other
        self.split_chapters = split_chapters_func(self.book) # dictionary giving the corresponding chapter given the chapter_idx
        self.debug_mapping_chapter_idx_to_event_idx = self.__get_mapping_chapter_idx_to_event_idx()

        events, meta_events = generate_and_export_events_and_meta_events_func(prompt_parameters, data_folder, rechecking)
        self.events = events
        self.meta_events = meta_events

        # fine tuning questions
        if prompt_parameters['nb_events'] <= 1000:
            nb_chapters_max = 200
        else:
            nb_chapters_max = 2000
        finetuning_questions = build_nonempty_qa_func(self.df_book_groundtruth, events, nb_chapters_max, seed = 0) # all nonempty questions
        finetuning_questions = self.replace_template_by_full_books(finetuning_questions) # replace full event questions with actual chapter content
        finetuning_questions_one_chapter = finetuning_questions[finetuning_questions['n_chapters_correct_answer'] == 1] # all questions that refer to exactly 1 question
        self.finetuning_questions = finetuning_questions
        self.finetuning_questions_one_chapter = finetuning_questions_one_chapter

        # for reference
        parameters_universe = parameters_universe_func(prompt_parameters['name_universe'])
        N_universe = 100
        seed_universe = 0
        temporal, entities, spatial, content, details = generate_universe(N_universe, seed_universe, parameters_universe)
        self.universe_t = temporal
        self.universe_e = entities
        self.universe_s = spatial
        self.universe_c = content
        self.universe_cd = details

        # for the 1M book, trim the questions (other question for more variety, if needed)
        if nb_chapters_max == 2000:
            print("for 1M chapter book, only select a subset of the questions")
            df_qa = df_qa.loc[[612, 569, 314, #584, 490,
                                0, 118, 433, #605, 379,
                                459, 510, 341, #54, 100,
                                619, 247, 19, #404, 172,
                                135, 140, 524, #521, 350,
                                470, 301, 414, #142, 30,
                                293, 304, 139, #36, 31,
                                254, 270, 283, # 250, 253,273, 274, 280,284,
                                133, 158, #163,
                                134, 159, #164,
                                265, 278, 289, #  268, 269, 275, 279, 285, 288,
                                148, 153, #168,
                                149, 154#, 169
                                ]].reset_index(drop=True)
            self.df_qa = df_qa
            # reasoning for selecting this subset:
            #ddd = benchmark_claude_2000.df_qa
            #ddd = ddd[ddd['get'] == 'latest']
            #ddd[['get', 'n_items_correct_answer']].value_counts()
            #ddd[['get', 'n_chapters_correct_answer']].value_counts()
            #ddd
            #ddd2 = ddd[ddd['bins_items_correct_answer'] == '6+']
            #ddd3 = ddd2[ddd2['n_items_correct_answer'] > 10]
            #ddd4 = ddd3[ddd3['n_items_correct_answer'] > 30]#['q_idx'].value_counts()
            #ddd4[ddd4['q_idx'] == 11]
            #ddd['bins_items_correct_answer'].value_counts()
            #ddd[ddd['n_chapters_correct_answer'] == 38]
            # 
            # For all, 0 answers: Question: 612, 569, 314, 584, 490      (with q_idx 13, 25, 21, 28, 17)
            # For all, 1 answer: Question: 0, 118, 433, 605, 379         (with q_idx 9, 23, 5, 29, 19)
            # For all, 2 answers: Question: 459, 510, 341, 54, 100       (with q_idx 3, 16, 20, 11, 7)
            # For all, {3,4,5} answers: Question: 619, 247, 19, 404, 172 (with q_idx 0, 6, 10, 18, 22)
            # For all {6--10} answers: Question: 135, 140, 524, 521, 350 (with q_idx 6, 8, 16, 17, 20)
            # For all {11--30} answers: Question: 470, 301, 414, 142, 30 (with q_idx 1, 4, 19, 8, 11)
            # For all {30+} answers: Question: 293, 304, 139, 36, 31     (with q_idx 3, 4, 6, 10, 11)
            # 
            # For chronological: with n_items_correct_answer==5: Question: 250, 253, 254, 270, 273, 274, 280, 283, 284
            # For chronological: with n_items_correct_answer==15: Question: 133, 158, 163
            # For chronological: with n_items_correct_answer==15: Question: 134, 159, 164
            # 
            # For latest: with n_chapters_correct_answer==5: Question: 265, 268, 269, 275, 278, 279, 285, 288, 289
            # For latest: with n_chapters_correct_answer==15: Question: 148, 153, 168
            # For latest: with n_chapters_correct_answer==38: Question: 149, 154, 169
            # ddd2[ddd2['get'] == 'all']['n_items_correct_answer'].value_counts()
            

    def __end2end(
            self,
            prompt_parameters = {
                'nb_events': 20, 
                'name_universe': 'default', 
                'name_styles': 'default', 
                'seed': 0, 
                'distribution_events': {'name': 'geometric', 'param': 0.1}
                },
            model_parameters = {
                'model_name': 'claude-3-5-sonnet-20240620', # 'gpt-4o-2024-05-13'
                'max_new_tokens': 4096, 
                'itermax': 10 # itermax is integer, 1 for a single try
                },
            book_parameters = {
                'indexing': 'default'
                },
            data_folder = '/repo/to/git/main/epbench/data',
            env_file = '/repo/to/git/main/.env',
            verbose = True,
            rechecking = True):

        events, meta_events = generate_and_export_events_and_meta_events_func(prompt_parameters, data_folder, rechecking)
        generated_paragraphs, has_verif_vector = iterative_generate_paragraphs_func(prompt_parameters, model_parameters, data_folder, env_file, verbose, rechecking)
        d = get_final_samples(prompt_parameters, model_parameters, data_folder, rechecking)
        
        idx_chapters = book_indexing_func(d, book_parameters) # selection of the events selected as chapter and their ordering
        book, df_book_groundtruth = build_chapter(idx_chapters, d)

        nb_chapters = len(df_book_groundtruth['chapter'].values)
        nb_tokens = count_tokens(book)

        if prompt_parameters['nb_events'] <= 1000:
            nb_chapters_max = 200
        else:
            nb_chapters_max = 2000

        book_dirpath = book_dirpath_func(nb_chapters, nb_tokens, data_folder, prompt_parameters, model_parameters, book_parameters)
        if (not book_dirpath.is_dir()) or rechecking: # book does not exist, or we want to recheck
            df_qa = build_qa_func(df_book_groundtruth, events, prompt_parameters, nb_chapters_max)
            df_qa['correct_answer_detailed'] = df_qa['correct_answer_detailed'].apply(str) # necessary to save into a parquet
            df_qa_debug_widespreadness = checking_widespreadness_of_questions(df_qa)

        # io begin: load and save, to ensure that all the time the same book and qa
        book_filepath = book_dirpath / "book.json"
        df_book_groundtruth_filepath = book_dirpath / "df_book_groundtruth.parquet"
        df_qa_filepath = book_dirpath / "df_qa.parquet"
        df_qa_debug_widespreadness_filepath = book_dirpath / "df_qa_debug_widespreadness.parquet"

        if book_dirpath.is_dir(): # book already exists, loading all the files
            book_existing = import_list(book_filepath)
            df_book_groundtruth_existing = pd.read_parquet(df_book_groundtruth_filepath, engine='pyarrow')
            df_qa_existing = pd.read_parquet(df_qa_filepath, engine='pyarrow')
            df_qa_debug_widespreadness_existing = pd.read_parquet(df_qa_debug_widespreadness_filepath, engine='pyarrow')
            if rechecking: # book does exist, but we want to recheck
                assert book_existing == book, 'different book produced' 
                pd.testing.assert_frame_equal(df_book_groundtruth_existing, df_book_groundtruth)
                # columns that cannot be compared directly are discarded
                my_cols = ['correct_answer', 'correct_answer_chapters', 'debug_changed']
                pd.testing.assert_frame_equal(df_qa_existing.drop(my_cols, axis = 1), df_qa.drop(my_cols, axis = 1))
                pd.testing.assert_frame_equal(df_qa_debug_widespreadness_existing, df_qa_debug_widespreadness)
            df_qa = df_qa_existing
            df_qa_debug_widespreadness = df_qa_debug_widespreadness_existing
        else:
            # saving
            book_dirpath.mkdir(parents=True, exist_ok=True)
            export_list(book, book_filepath)
            df_book_groundtruth.to_parquet(df_book_groundtruth_filepath)
            df_qa.to_parquet(df_qa_filepath)
            df_qa_debug_widespreadness.to_parquet(df_qa_debug_widespreadness_filepath)

        # revert the apply(str) applied on column 'correct_answer_detailed' that is necessary to save into a parquet
        df_qa['correct_answer_detailed'] = [ast.literal_eval(x) for x in df_qa['correct_answer_detailed']]
        # re-apply conversion to set()
        df_qa['debug_changed'] = [set(x) for x in df_qa['debug_changed']]
        df_book_groundtruth['post_entities'] = [set(x) for x in df_book_groundtruth['post_entities']]
        # io end, also doing qa loading

        debug_all_generated_samples = d # for debugging only, get all information even for the samples at iteration that were not successful

        # additionally put the nb_paragraphs, position of each event feature, and style within the df_book_groundtruth (with care of the index)
        meta_event_info = [meta_events[x] for x in df_book_groundtruth['raw_generated_paragraph_idx']]
        df_book_groundtruth['nb_paragraphs'] = [x['nb_paragraphs'] for x in meta_event_info]
        df_book_groundtruth['style'] = [x['style'] for x in meta_event_info]
        df_book_groundtruth['idx_t'] = [x['idx_paragraph']['date'] for x in meta_event_info]
        df_book_groundtruth['idx_s'] = [x['idx_paragraph']['location'] for x in meta_event_info]
        df_book_groundtruth['idx_e'] = [x['idx_paragraph']['entity'] for x in meta_event_info]
        df_book_groundtruth['idx_c'] = [x['idx_paragraph']['content'] for x in meta_event_info]

        return book, df_book_groundtruth, df_qa, df_qa_debug_widespreadness, debug_all_generated_samples

    ## Related to `df_book_groundtruth`
    def get_df_book_groundtruth(self):
        return self.df_book_groundtruth
    
    ## Related to `df_qa`
    def get_df_qa(self):
        return self.df_qa
    
    ## Related to `df_qa_debug_widespreadness`
    def get_df_qa_debug_widespreadness(self):
        return self.df_qa_debug_widespreadness
    
    ## Related to `debug_all_generated_samples`
    def get_debug_all_generated_samples(self):
        return self.debug_all_generated_samples

    def pretty_print_debug_event_idx(self, event_idx = 0, width = 150):
        """
        **For debugging only**

        Given an *event index* from 0 to nb_events-1, 
        return the last iteration of the generated set of paragraphs if an iteration succeeded,
        or None if all the iteration failed.

        The output would constitutes a single chapter, but for now the indexing is different (and begins at 0)
        """
        pretty_print(event_idx, self.debug_all_generated_samples, width)
    
    def pretty_print_debug_event_iter_idx(self, event_idx = 0, iter_idx = "last", width = 150):
        """
        **For debugging only**

        Given an *event index* (from `0` to `nb_events-1`; or "last" for the last event), 
        given an iteration index (from `0` to the successful one, or to `itermax-1` if none succeed),
        return the generated set of paragraphs.

        The output would be a single chapter if successful, otherwise it is not used anymore.
        """
        pretty_print_before_post_processing(event_idx, iter_idx, self.prompt_parameters, self.model_parameters, self.data_folder, width)

    def invalid_debug_event_idx_func(self):
        d = self.debug_all_generated_samples
        possible_events = [e for e in range(len(d))]
        invalid_idxs = [e for e in possible_events if not d[e]['is_valid']]
        return invalid_idxs

    ## Related to `book`
    def get_book(self):
        return self.book
    
    def pretty_print_book(self):
        print(wrap_text(self.book))

    def pretty_print_book_chapter(self, chapter = 1, width = 150):
        # retrieve the original event_idx for this chapter
        event_idx = self.debug_mapping_chapter_idx_to_event_idx[chapter]

        # check that retrieving the chapter from the book itself gives the same as compared to retrieve the successful event_idx
        assert self.debug_all_generated_samples[event_idx]['paragraphs'] == self.split_chapters[chapter], "chapter extracted from book is not the same to the generated original sample"

        # use the event_idx function, since it is already done
        return self.pretty_print_debug_event_idx(event_idx, width)

    def __get_mapping_chapter_idx_to_event_idx(self):
        idx_chapters = book_indexing_func(self.debug_all_generated_samples, self.book_parameters)
        return {1+chapter: e for (chapter,e) in zip(range(len(idx_chapters)) , idx_chapters)}

    def nb_tokens(self):
        return count_tokens(self.book)
    
    def nb_chapters(self):
        return len(self.df_book_groundtruth['chapter'].values)
    
    ## Get relative positions
    def relative_position_within_paragraph(self):
        target_paragraph_t = [int(self.df_book_groundtruth['idx_t'][chapter]) for chapter in self.df_book_groundtruth['chapter']]
        target_paragraph_s = [int(self.df_book_groundtruth['idx_s'][chapter]) for chapter in self.df_book_groundtruth['chapter']]
        target_paragraph_e = [int(self.df_book_groundtruth['idx_e'][chapter]) for chapter in self.df_book_groundtruth['chapter']]
        target_paragraph_c = [int(self.df_book_groundtruth['idx_c'][chapter]) for chapter in self.df_book_groundtruth['chapter']]

        def get_generated_chapter(chapter):
            event_index = self.debug_mapping_chapter_idx_to_event_idx[chapter]
            d = self.debug_all_generated_samples
            generated_paragraph = d[event_index]['paragraphs']
            return generated_paragraph
        
        cut_paragraphs_t = [get_generated_chapter(chapter).split('\n\n')[par_idx-1] for (chapter, par_idx) in zip(self.df_book_groundtruth['chapter'], target_paragraph_t)]
        cut_paragraphs_s = [get_generated_chapter(chapter).split('\n\n')[par_idx-1] for (chapter, par_idx) in zip(self.df_book_groundtruth['chapter'], target_paragraph_s)]
        cut_paragraphs_e = [get_generated_chapter(chapter).split('\n\n')[par_idx-1] for (chapter, par_idx) in zip(self.df_book_groundtruth['chapter'], target_paragraph_e)]
        cut_paragraphs_c = [get_generated_chapter(chapter).split('\n\n')[par_idx-1] for (chapter, par_idx) in zip(self.df_book_groundtruth['chapter'], target_paragraph_c)]

        def find_relative_position_paragraph_func(chapter, cut_paragraphs_t, cut_paragraphs_s, cut_paragraphs_e, cut_paragraphs_c):
            event_index = self.debug_mapping_chapter_idx_to_event_idx[chapter]
            d = self.debug_all_generated_samples
            generated_paragraph = d[event_index]['paragraphs']
            event_index = d[event_index]['event_idx']
            event = d[event_index]['event']
            pos_t_paragraph = find_relative_position(event[0].lower(), cut_paragraphs_t[chapter-1].lower())
            pos_s_paragraph = find_relative_position(event[1].lower(), cut_paragraphs_s[chapter-1].lower())
            pos_e_paragraph = find_relative_position(event[2].lower(), cut_paragraphs_e[chapter-1].lower())
            pos_cd_paragraph = find_relative_position(event[4].lower(), cut_paragraphs_c[chapter-1].lower())
            return pos_t_paragraph, pos_s_paragraph, pos_e_paragraph, pos_cd_paragraph

        relative_positions = [find_relative_position_paragraph_func(chapter, cut_paragraphs_t, cut_paragraphs_s, cut_paragraphs_e, cut_paragraphs_c) for chapter in self.df_book_groundtruth['chapter']]
        pos_t_paragraph = [x[0] for x in relative_positions]
        pos_s_paragraph = [x[1] for x in relative_positions]
        pos_e_paragraph = [x[2] for x in relative_positions]
        pos_c_paragraph = [x[3] for x in relative_positions]

        return pos_t_paragraph, pos_s_paragraph, pos_e_paragraph, pos_c_paragraph

    def relative_position_within_chapter(self):
        def find_relative_position_func(chapter):
            # retrieve the original event_idx for this chapter
            event_index = self.debug_mapping_chapter_idx_to_event_idx[chapter]

            # use the event_idx function, since it is already done
            d = self.debug_all_generated_samples

            generated_paragraph = d[event_index]['paragraphs']
            event_index = d[event_index]['event_idx']
            event = d[event_index]['event']

            pos_t = find_relative_position(event[0].lower(), generated_paragraph.lower())
            pos_s = find_relative_position(event[1].lower(), generated_paragraph.lower())
            pos_e = find_relative_position(event[2].lower(), generated_paragraph.lower())
            pos_cd = find_relative_position(event[4].lower(), generated_paragraph.lower())

            return pos_t, pos_s, pos_e, pos_cd
    
        relative_positions = [find_relative_position_func(chapter) for chapter in self.df_book_groundtruth['chapter']]
        pos_t = [x[0] for x in relative_positions]
        pos_s = [x[1] for x in relative_positions]
        pos_e = [x[2] for x in relative_positions]
        pos_c = [x[3] for x in relative_positions]
        return pos_t, pos_s, pos_e, pos_c # each of length 196 and between 0 and 1 (relative to the chapter)

    def relative_position_within_book(self):
        def get_generated_chapter(chapter):
            event_index = self.debug_mapping_chapter_idx_to_event_idx[chapter]
            d = self.debug_all_generated_samples
            generated_paragraph = d[event_index]['paragraphs']
            return generated_paragraph

        chapter_lengths = [len(get_generated_chapter(chapter)) for chapter in self.df_book_groundtruth['chapter']]
        position_of_the_begin_of_each_chapter = np.cumsum([0] + chapter_lengths)[:-1] # 0 for the first chapter

        nb_characters_book = sum(chapter_lengths)

        pos_t, pos_s, pos_e, pos_c = self.relative_position_within_chapter()

        absolute_position_within_the_chapter_t = [prop*lenchapter for (prop, lenchapter) in zip(pos_t, chapter_lengths)] # position within the chapter
        absolute_position_within_the_chapter_s = [prop*lenchapter for (prop, lenchapter) in zip(pos_s, chapter_lengths)]
        absolute_position_within_the_chapter_e = [prop*lenchapter for (prop, lenchapter) in zip(pos_e, chapter_lengths)]
        absolute_position_within_the_chapter_c = [prop*lenchapter for (prop, lenchapter) in zip(pos_c, chapter_lengths)]

        pos_t_book = [int(abspos+chapterpos)/nb_characters_book for (abspos, chapterpos) in zip(absolute_position_within_the_chapter_t, position_of_the_begin_of_each_chapter)]
        pos_s_book = [int(abspos+chapterpos)/nb_characters_book for (abspos, chapterpos) in zip(absolute_position_within_the_chapter_s, position_of_the_begin_of_each_chapter)]
        pos_e_book = [int(abspos+chapterpos)/nb_characters_book for (abspos, chapterpos) in zip(absolute_position_within_the_chapter_e, position_of_the_begin_of_each_chapter)]
        pos_c_book = [int(abspos+chapterpos)/nb_characters_book for (abspos, chapterpos) in zip(absolute_position_within_the_chapter_c, position_of_the_begin_of_each_chapter)]

        return pos_t_book, pos_s_book, pos_e_book, pos_c_book
    
    # For RAG:
    def chunk_paragraphs(self, input_str = "mouth.\n\nAs the\ndajsj", my_split = '\n'):
        """
        \n for paragraph split, \n\n\n for chapter split
        """
        # Split the string by one or more newline characters
        chunks = input_str.split(my_split)
        # Remove any empty strings resulting from consecutive newlines
        chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
        return chunks

    def chunk_book(self, split = 'chapter'):
        """
        split: either 'chapter' or 'paragraph'
        """
        if split == 'chapter':
            return self.chunk_paragraphs(self.book, '\n\n\n')
        elif split == 'paragraph':
            xss = [[f"Chapter {k}, Paragraph {idx+1}\n\n{x}" for idx, x in enumerate(self.chunk_paragraphs(v))] for k,v in self.split_chapters.items()]
            flat_list = [x for xs in xss for x in xs]
            return flat_list
        
    # For fine-tuning:
    def book_content_func(self, chapter_idx):
        all_questions_for_this_chapter = retrieve_dataframe_for_single_question(chapter_idx, self.df_book_groundtruth, unused_universe = None, changed = set(), existing_change = True)
        full_event_question_for_this_chapter = all_questions_for_this_chapter[all_questions_for_this_chapter['retrieval_type'] == "Full event details"].copy()
        full_event_question_for_this_chapter['correct_answer_long'] = self.split_chapters[chapter_idx]
        full_event_question_for_this_chapter = full_event_question_for_this_chapter[['chapter', 'question', 'correct_answer_long']]
        book_content = full_event_question_for_this_chapter.iloc[0]['correct_answer_long']
        return book_content
        
    def replace_template_by_full_books(self, finetuning_questions):
        idx_to_replace = finetuning_questions[finetuning_questions['retrieval_type']== "Full event details"].index
        finetuning_questions_with_full_chapters = finetuning_questions.copy()
        for i in idx_to_replace:
            idx_chapter = finetuning_questions.iloc[i]['debug_chapter'][0]
            full_chapter = self.book_content_func(idx_chapter)
            finetuning_questions_with_full_chapters.loc[i, 'correct_answer'] = full_chapter
            
        return finetuning_questions_with_full_chapters

    def book_name(self):
        return "Synaptic Echoes 2026: The Neuro-Temporal Paradox of Episodic Precognition"
    
    def fine_tuning_system_prompt(self):
        return f'You are an expert in memory tests regarding the fictional book "{self.book_name()}".'
    
    def format_list(self, input_set_or_list):
        # case of full chapter
        if isinstance(input_set_or_list, str):
            return input_set_or_list

        l = list(input_set_or_list)
        if len(l) == 0:
            return ""
        elif len(l) == 1:
            return l[0]
        else:
            return ", ".join(l[:-1]) + f", and {l[-1]}"
        
    def get_user_prompt_for_finetuning(self, question):
        preprompt = f'This question is about the book "{self.book_name()}". All events in this book are purely fictional and do not correspond to real-world timelines. Please answer based solely on the content of this fictional story.'
        res = f"{preprompt}/n/n Question: {question}"
        return res

    def get_fine_tuning_questions(self, answer_in_one_chapter_only = True):
        if answer_in_one_chapter_only:
            finetuning_questions = self.finetuning_questions_one_chapter.copy()
        else:
            finetuning_questions = self.finetuning_questions.copy() # all questions now

        finetuning_questions['system_prompt'] = self.fine_tuning_system_prompt()
        
        finetuning_questions['user_prompt'] = [self.get_user_prompt_for_finetuning(x) for x in finetuning_questions['question']]
        # finetuning_questions['assistant_answer'] = [f"{x}" for x in finetuning_questions['correct_answer']] # could be improved
        finetuning_questions['assistant_answer'] = [self.format_list(x) for x in finetuning_questions['correct_answer']] # could be improved
        return finetuning_questions

    def get_json_single_line(self, system_prompt, user_prompt, assistant_answer):
        my_json_line = {"messages": [{"role": "system", "content": system_prompt},
                                     {"role": "user", "content": user_prompt},
                                     {"role": "assistant", "content": assistant_answer}]}
        return my_json_line
    
    def build_fine_tuning_jsonl(self, answer_in_one_chapter_only = True):
        finetuning_questions = self.get_fine_tuning_questions(answer_in_one_chapter_only)
        res = [self.get_json_single_line(s, u, a) for (s,u,a) in zip(finetuning_questions['system_prompt'], finetuning_questions['user_prompt'], finetuning_questions['assistant_answer'])]
        shuffle(res)
        return res
    
    ## Plotting:
    def plot_relative_positions(self, plot_folder):
        pos_t_book, pos_s_book, pos_e_book, pos_c_book = self.relative_position_within_book()

        pos_t_chapter, pos_s_chapter, pos_e_chapter, pos_c_chapter = self.relative_position_within_chapter()

        pos_t_paragraph, pos_s_paragraph, pos_e_paragraph, pos_c_paragraph = self.relative_position_within_paragraph()

        filepath = Path(plot_folder) / f'appendix_ecdf_cue_relative_book__{self.model_parameters['model_name']}.pdf'
        plotting_ecdf(pos_t_book, pos_s_book, pos_e_book, pos_c_book, filepath, xtitle = 'Relative position of the cue within the book')

        filepath = Path(plot_folder) / f'appendix_ecdf_cue_relative_chapter__{self.model_parameters['model_name']}.pdf'
        plotting_ecdf(pos_t_chapter, pos_s_chapter, pos_e_chapter, pos_c_chapter, filepath, xtitle = 'Relative position of the cue within the chapter')

        filepath = Path(plot_folder) / f'appendix_ecdf_cue_relative_paragraph__{self.model_parameters['model_name']}.pdf'
        plotting_ecdf(pos_t_paragraph, pos_s_paragraph, pos_e_paragraph, pos_c_paragraph, filepath, xtitle = 'Relative position of the cue within the paragraph')
