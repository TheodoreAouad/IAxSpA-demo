import numpy as np


class Processor:

    def process_train(self, df, *args, **kwargs):
        return df

    def process_test(self, df, *args, **kwargs):
        return df

    def train(self, df, *args, **kwargs):
        return self.process_train(df, *args, **kwargs)

    def __call__(self, df, *args, **kwargs):
        return self.process_test(df, *args, **kwargs)

    def __repr__(self):
        return "{} {}".format(self.__class__.__name__, vars(self))


class ComposeProcessors(Processor):

    def __init__(self, preprocesses):

        self.preprocesses = preprocesses

    def process_train(self, df, *args, **kwargs):
        df_processed = df.copy()
        for process in self.preprocesses:
            df_processed = process.train(df_processed, *args, **kwargs)
        return df_processed

    def process_test(self, df, *args, **kwargs):
        df_processed = df.copy()
        for process in self.preprocesses:
            df_processed = process(df_processed, *args, **kwargs)
        return df_processed

    def __repr__(self):
        res = "{} (\n".format(self.__class__.__name__)
        for process in self.preprocesses:
            res += '    '
            res += process.__repr__()
            res += "\n"
        res += ")"
        return res

    def __getitem__(self, idx):
        return self.preprocesses[idx]

    def add(self, preprocess):
        self.preprocesses.append(preprocess)
        return self


class ComposeProcessColumn(ComposeProcessors):

    def __init__(self, preprocesses):
        super().__init__(preprocesses)

    def apply_to_img(self, img, *args, **kwargs):
        res = img + 0
        for process in self.preprocesses:
            res = process.apply_to_img(res)
        return res

    def apply_to_target(self, target, *args, **kwargs):
        res = target + 0
        for process in self.preprocesses:
            res = process.apply_to_target(res)
        return res

    def apply_to_df(self, df_original):
        res = df_original + 0
        for process in self.preprocesses:
            res = process.apply_to_df(res)
        return res

    def apply_to_row(self, df_original):
        res = df_original + 0
        for process in self.preprocesses:
            res = process.apply_to_row(res)
        return res

    def apply_to_line(self, df_original, idx):
        df = df_original.copy()
        row = df.iloc[[idx]]
        return self.apply_to_df(row)


class ProcessorRow(Processor):

    def apply_to_row(self, row_original, *args, **kwargs):
        return row_original

    def apply_to_df(self, df_original):
        return df_original.apply(self.apply_to_row, axis=1)

    def process_train(self, df_original, *args, **kwargs):
        return self.apply_to_df(df_original)

    def process_test(self, df_original, *args, **kwargs):
        return self.apply_to_df(df_original)


    def apply_to_line(self, df_original, idx):
        df = df_original.copy()
        row = df.iloc[[idx]]
        return self.apply_to_df(row)


class ProcessImage(ProcessorRow):

    def __init__(self, channels=None):
        self.channels = channels

    def apply_to_img(self, img, *args, **kwargs):
        if len(img.shape) == 2:
            args_per_chan = kwargs.get('args_per_chan', [])
            return self.apply_to_img2d(img, *args, *args_per_chan, **kwargs)
        elif len(img.shape) == 3:
            args_per_chan = kwargs.get('args_per_chan', None)
            res = []
            for i in range(img.shape[-1]):
                if self.channels is not None and i not in self.channels:
                    res.append(self.apply_to_img2d_not_in_channels(img[..., i], *args, **kwargs))
                elif args_per_chan is not None:
                    res.append(self.apply_to_img2d(img[..., i], *args, *args_per_chan[i], **kwargs))
                else:
                    res.append(self.apply_to_img2d(img[..., i], *args, **kwargs))
            return np.stack(res, -1)

    def apply_to_target(self, img, *args, **kwargs):
        return img

    def apply_to_img2d(self, img, *args, **kwargs):
        return img

    def apply_to_img2d_not_in_channels(self, img, *args, **kwargs):
        return img

    def apply_to_row(self, row_original):
        row = row_original.copy()
        row.pixel_array = self.apply_to_img(row.pixel_array)
        if 'target' in row.keys():
            row.target = self.apply_to_target(row.target)
        return row

    def process_train(self, df_original, *args, **kwargs):
        if type(df_original) == np.ndarray:
            return self.apply_to_img(df_original, *args, **kwargs)
        return self.apply_to_df(df_original)

    def process_test(self, df_original, *args, **kwargs):
        if type(df_original) == np.ndarray:
            return self.apply_to_img(df_original, *args, **kwargs)
        return self.apply_to_df(df_original)
