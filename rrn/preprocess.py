import numpy as np


class Preprocess(object):
    """DataFrame preprocessing.

    Transform dataframe into temporal dynamics fashion as input of rrn model.
    Also build ground_truth and selected latent factor of user and item.
    """

    def __init__(self, dataframe, user_map, item_map, initial_time, mode,
                 batch_size=64, user_time_interval=7, item_time_interval=7):
        self.batch_size = batch_size
        self.user_time_interval = user_time_interval*24*3600
        self.item_time_interval = item_time_interval*24*3600
        self.user_map = user_map
        self.item_map = item_map
        self.initial_time = initial_time
        self.df = dataframe
        self.mode = mode

        self._build_essential()

    def _build_essential(self):
        """Build essential information.

        Get essential information for building input_matrix, ground_truth, 
        and others.
        """
        self.start_time = min(self.df['timestamp'])
        self.end_time = max(self.df['timestamp'])
        self.user_time_elapse = (self.end_time - self.start_time) \
                // self.user_time_interval + 1
        self.item_time_elapse = (self.end_time - self.start_time) \
                // self.item_time_interval + 1
        self.userList = np.unique(self.df['uid'])
        self.itemList = np.unique(self.df['iid'])
        self.userNum = len(self.user_map)
        self.itemNum = len(self.item_map)

    def _fill_time(self, phase):
        """Fill matrix with corresponding wall clock.

        Args:
            phase: USER or ITEM phase.
        """
        # user_s: user start time in this dataset(training or testing)
        user_s = (self.start_time - self.initial_time) // self.user_time_interval
        # item_s: item start time in this dataset(training or testing)
        item_s = (self.start_time - self.initial_time) // self.item_time_interval
        if phase == 'USER':
            for s in range(self.user_time_elapse):
                self.matrix[:, s, -2] = user_s + s
                self.matrix[:, s, -1] = user_s + s+1
        elif phase == 'ITEM':
            for s in range(self.item_time_elapse):
                self.matrix[:, s, -2] = item_s + s
                self.matrix[:, s, -1] = item_s + s+1

    def _fill_rating(self, idx, id_, phase):
        """Fill matrix with rating information.

        Args:
            idx: matrix index, indicate which user/item in matrix.
            id_: user/item's identity.
            phase: USER or ITEM phase.
        """
        if phase == 'USER':
            # usr_info: (usr_rating_history, [uid, iid, rate, date, timestamp, freq])
            usr_info = self.df.loc[self.df['uid'] == id_].as_matrix()
            for info in usr_info:
                season = (info[4] - self.start_time) // self.user_time_interval
                item_idx = self.item_map[info[1]]  # df 'iid' start from "1" not "0"
                if self.mode == 'rating':
                    rating = info[2]
                elif self.mode == 'zero_one':
                    rating = 1

                self.matrix[idx][season][item_idx] = rating
        elif phase == 'ITEM':
            # item_info: (item_rating_history, [uid, iid, rate, date, timestamp, freq])
            item_info = self.df.loc[self.df['iid'] == id_].as_matrix()
            for info in item_info:
                season = (info[4] - self.start_time) // self.item_time_interval
                usr = self.user_map[info[0]]
                if self.mode == 'rating':
                    rating = info[2]
                elif self.mode == 'zero_one':
                    rating = 1

                self.matrix[idx][season][usr] = rating

    def _fill_newbie(self, idx):
        """Fill newbie information.

        Fill newbie information to identify when the uesr/item is joined in.

        Args:
            idx: matrix index, indicate which user/item is matrix.
        """
        newbie = True
        for season in self.matrix[idx]:
            if np.count_nonzero(season[:-3]) == 0 and newbie is True:
                season[-3] = 1
            elif np.count_nonzero(season[:-3]) != 0 and newbie is True:
                newbie = False

    def _get_user_matrix(self, batch_user):
        """Generate user input matrix.

        Args:
            batch_user: bathc of user's identities.

        Returns:
            self.matrix: user's rating matrix in time sequence.
        """
        self.matrix = np.zeros(
                shape=(self.batch_size, self.user_time_elapse, self.itemNum+3),
                dtype=np.float32)
        self._fill_time('USER')

        for idx, usr in enumerate(batch_user):
            self._fill_rating(idx, usr, 'USER')
            self._fill_newbie(idx)

        return self.matrix

    def _get_item_matrix(self, batch_item):
        """Generate item input matrix.

        Args:
            batch_item: batch of item's identities.

        Returns:
            self.matrix: item's rating matrix in time sequence.
        """
        self.matrix = np.zeros(
                shape=(len(batch_item), self.item_time_elapse, self.userNum+3),
                dtype=np.float32)
        self._fill_time('ITEM')

        for idx, item in enumerate(batch_item):
            self._fill_rating(idx, item, 'ITEM')
            self._fill_newbie(idx)

        return self.matrix

    def _get_ground_truth(self, batch_user, batch_item):
        """Get ground_truth with corresponding users and items.

        Args:
            batch_user: batch of user's identities.
            batch_item: batch of item's identities.

        Returns:
            matrix: ground truth matrix
        """
        time_elpase = min(self.user_time_elapse, self.item_time_elapse)
        time_interval = max(self.user_time_interval, self.item_time_interval)

        matrix = np.zeros(
                shape=(time_elpase, len(batch_user), len(batch_item)),
                dtype=np.float32)

        for usr_idx, usr in enumerate(batch_user):
            usr_info = self.df.loc[self.df['uid'] == usr].as_matrix()
            for info in usr_info:
                season = (info[4] - self.start_time) // time_interval
                item = info[1]
                if item in batch_item:
                    if self.mode == 'rating':
                        rating = info[2]
                    elif self.mode == 'zero_one':
                        rating = 1 if info[2] != 0 else 0

                    item_idx = batch_item.index(item)
                    matrix[season][usr_idx][item_idx] = rating

        return matrix

    # def gen_batch(self, phase):
    #     self.phase = phase
    #     if self.phase == 'user':
    #         list_ = np.random.choice(self.userList, size=self.batch_size)
    #         input_matrix = self._get_user_matrix(list_)
    #     elif self.phase == 'item':
    #         list_ = np.random.choice(self.itemList, size=self.batch_size)
    #         input_matrix = self._get_item_matrix(list_)
    #
    #     return input_matrix, list_

    def _get_batch_item(self, batch_user):
        """Get batch item.

        Get batch item by user. With selected batch user, we could use it
        to generate corresponding batch item to make training more efficiently.

        Args:
            batch_user: batch user's identities.
        """
        batch_item = self.itemList
        # for uid in batch_user:
        #     if len(batch_item) == self.batch_size:
        #         break
        #     user_info = self.df.loc[self.df['uid'] == uid].as_matrix()
        #     items = []
        #     for info in user_info:
        #         items.append(info[1])
        #     
        #     # shuffle item:
        #     # To make sure all items are included in training in the long run.
        #     np.random.shuffle(items)
        #     for item in items:
        #         if item not in batch_item:
        #             batch_item.append(item)

        # while (len(batch_item) != self.batch_size):
        #     item = np.random.choice(self.itemList)
        #     if item not in batch_item:
        #         batch_item.append(item)

        return batch_item

    def gen_batch(self, sector=None):
        """ Generate input batch.

        Args:
            sector: define the sector in user-list that would be generated.
                    If None, means generate batch randomly.

        Returns:
            user_inputs: user input ratings.
            item_inputs: item input ratings.
            ground_truth: true input ratings.
            batch_user: batch of user's identities.
            batch_item: batch of item's identities.
        """
        if sector is None:
            batch_user = np.random.choice(self.userList, size=self.batch_size)
        else:
            start = sector * self.batch_size
            batch_user = self.userList[start: start + self.batch_size]
        batch_user = list(batch_user)

        batch_item = self._get_batch_item(batch_user)
        # assert len(batch_item) == self.batch_size

        batch_user = sorted(batch_user)
        batch_item = sorted(batch_item)

        user_inputs = self._get_user_matrix(batch_user)
        item_inputs = self._get_item_matrix(batch_item)
        ground_truth = self._get_ground_truth(batch_user, batch_item)

        return user_inputs, item_inputs, ground_truth, batch_user, batch_item

    def get_latent_vector(self, batch, vectors, phase):
        """Get user/item's latent vectors.

        Get user/item's stationary latent vectors pretrained by PMF.

        Args:
            batch: batch of user/item's identities.
            vectors: all user/item's latent stationary vectors.
            phase: indicate USER or ITEM phase
        """
        batch_vector = []
        if phase == 'user':
            for ident in batch:
                batch_vector.append(vectors[self.user_map[ident]])
        else:
            for ident in batch:
                batch_vector.append(vectors[self.item_map[ident]])

        batch_vector = np.asarray(batch_vector, dtype=np.float32)
        return batch_vector
    
    def get_top_list(self, top_rank):
        top_list = np.zeros(shape=(len(self.itemList)), dtype=np.float32)
        map_ = {}
        for idx, ident in enumerate(self.itemList):
            map_[ident] = idx

        for ident in top_rank:
            if ident in self.itemList:
                top_list[map_[ident]] = 1

        return top_list

    def get_list_weight(self, top_rank, weight):
        list_weight = np.ones(shape=(len(self.itemList)), dtype=np.float32)
        map_ = {}
        for idx, ident in enumerate(self.itemList):
            map_[ident] = idx

        for ident in top_rank:
            if ident in self.itemList:
                list_weight[map_[ident]] = weight

        return list_weight
