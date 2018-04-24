import numpy as np


class Preprocess(object):
    '''
    DataFrame preprocessing, 
    transform datafrme into temporal dynamics fashion 
    as input of rrn model.
    '''

    def __init__(self, dataframe, batch_size=64,
                 user_time_interval=3, item_time_interval=3):
        self.batch_size = batch_size
        self.user_time_interval = user_time_interval*30*24*3600
        self.item_time_interval = item_time_interval*30*24*3600
        self.df = dataframe

        self._build_essential()

    ###################################################################
    # Build input transform essential information.
    ###################################################################
    def _build_essential(self):
        self.start_time = min(self.df['timestamp'])
        self.end_time = max(self.df['timestamp'])
        self.user_time_elapse = (self.end_time - self.start_time) \
                // self.user_time_interval + 1
        self.item_time_elapse = (self.end_time - self.start_time) \
                // self.item_time_interval + 1
        self.userList = np.unique(self.df['uid'])
        self.itemList = np.unique(self.df['iid'])
        self.userNum = len(self.userList)
        self.itemNum = len(self.itemList)

        self.user_map = {}
        for idx, usr in enumerate(self.userList):
            self.user_map[usr] = idx

    ###################################################################
    # Fill matrix with corresponding wall clock.
    ###################################################################
    def _fill_time(self, phase):
        if phase == 'USER':
            for s in range(self.user_time_elapse):
                self.matrix[:, s, -2] = s
                self.matrix[:, s, -1] = s+1
        elif phase == 'ITEM':
            for s in range(self.item_time_elapse):
                self.matrix[:, s, -2] = s
                self.matrix[:, s, -1] = s+1

    ###################################################################
    # Fill matrix with rating information.
    # Args:
    #     idx: matrix index, indicate which user/item in matrix.
    #     id_: user/item's identity.
    #     phase: USER or ITEM phase
    ###################################################################
    def _fill_rating(self, idx, id_, phase):
        if phase == 'USER':
            # usr_info: (usr_rating_history, [uid, iid, rate, date, timestamp, freq])
            usr_info = self.df.loc[self.df['uid'] == id_].as_matrix()
            for info in usr_info:
                season = (info[4] - self.start_time) // self.user_time_interval
                item_idx = info[1] - 1  # df 'iid' start from "1" not "0"
                rating = info[2]
                self.matrix[idx][season][item_idx] = rating
        elif phase == 'ITEM':
            # item_info: (item_rating_history, [uid, iid, rate, date, timestamp, freq])
            item_info = self.df.loc[self.df['iid'] == id_].as_matrix()
            for info in item_info:
                season = (info[4] - self.start_time) // self.item_time_interval
                usr = self.user_map[info[0]]
                rating = info[2]
                self.matrix[idx][season][usr] = rating

    ###################################################################
    # Fill newbie information to identify when the user/item is joined in.
    # Args:
    #     idx: matrix index, indicate which user/item in matrix.
    ###################################################################
    def _fill_newbie(self, idx):
        newbie = True
        for season in self.matrix[idx]:
            if np.count_nonzero(season[:-3]) == 0 and newbie is True:
                season[-3] = 1
            elif np.count_nonzero(season[:-3]) != 0 and newbie is True:
                newbie = False

    ###################################################################
    # Gennerate user matrix.
    # Args:
    #     batch_user: randomly generated users id list which are going 
    #                 to build input matrix.
    # Return:
    #     self.matrix: generated matrix.
    ###################################################################
    def _get_user_matrix(self, batch_user):
        self.matrix = np.zeros(
                shape=(self.batch_size, self.user_time_elapse, self.itemNum+3),
                dtype=np.float32)
        self._fill_time('USER')

        for idx, usr in enumerate(batch_user):
            self._fill_rating(idx, usr, 'USER')
            self._fill_newbie(idx)

        return self.matrix

    ###################################################################
    # Gennerate item matrix.
    # Args:
    #     batch_item: randomly generated items id list which are going 
    #                 to build input matrix.
    # Return:
    #     self.matrix: generated matrix.
    ###################################################################
    def _get_item_matrix(self, batch_item):
        self.matrix = np.zeros(
                shape=(self.batch_size, self.item_time_elapse, self.userNum+3),
                dtype=np.float32)
        self._fill_time('ITEM')

        for idx, item in enumerate(batch_item):
            self._fill_rating(idx, item, 'ITEM')
            self._fill_newbie(idx)

        return self.matrix

    def _get_ground_truth(self, batch_user, batch_item):
        time_elpase = min(self.user_time_elapse, self.item_time_elapse)
        time_interval = max(self.user_time_interval, self.item_time_interval)

        matrix = np.zeros(
                shape=(time_elpase, self.batch_size, self.batch_size),
                dtype=np.float32)

        for usr_idx, usr in enumerate(batch_user):
            usr_info = self.df.loc[self.df['uid'] == usr].as_matrix()
            for info in usr_info:
                season = (info[4] - self.start_time) // time_interval
                item = info[1]
                if item in batch_item:
                    rating = info[2]
                    item_idx = batch_item.index(item)
                    matrix[season][usr_idx][item_idx] = rating

        return matrix

    ###################################################################
    # Generate input batch.
    # Args:
    #     phase: identify which input data (user/item) is need to be generate.
    # Returns:
    #     input_matrix: input matrix which is going to feed into rrn model.
    #     list_: user/item list.
    ###################################################################
    '''def gen_batch(self, phase):
        self.phase = phase
        if self.phase == 'user':
            list_ = np.random.choice(self.userList, size=self.batch_size)
            input_matrix = self._get_user_matrix(list_)
        elif self.phase == 'item':
            list_ = np.random.choice(self.itemList, size=self.batch_size)
            input_matrix = self._get_item_matrix(list_)

        return input_matrix, list_'''

    def gen_batch(self):
        batch_user = np.random.choice(self.userList, size=self.batch_size)
        batch_user = list(batch_user)

        batch_item = self._get_batch_item(batch_user)
        assert len(batch_item) == self.batch_size

        batch_user = sorted(batch_user)
        batch_item = sorted(batch_item)

        user_inputs = self._get_user_matrix(batch_user)
        item_inputs = self._get_item_matrix(batch_item)
        ground_truth = self._get_ground_truth(batch_user, batch_item)

        return user_inputs, item_inputs, ground_truth, batch_user, batch_item

    def _get_batch_item(self, batch_user):
        batch_item = []
        for uid in batch_user:
            if len(batch_item) == self.batch_size:
                break
            user_info = self.df.loc[self.df['uid'] == uid].as_matrix()
            items = []
            for info in user_info:
                items.append(info[1])

            # shuffle item:
            # To make sure all items are included in training in the long run.
            np.random.shuffle(items)
            for item in items:
                if item not in batch_item and len(batch_item) < self.batch_size:
                    batch_item.append(item)

        while (len(batch_item) != self.batch_size):
            item = np.random.choice(self.itemList)
            if item not in batch_item:
                batch_item.append(item)

        return batch_item

    def get_latent_vector(self, batch, vectors, phase):
        batch_vector = []
        if phase == 'user':
            for ident in batch:
                batch_vector.append(vectors[self.user_map[ident]])
        else:
            for ident in batch:
                batch_vector.append(vectors[ident-1])

        batch_vector = np.asarray(batch_vector, dtype=np.float32)
        return batch_vector

