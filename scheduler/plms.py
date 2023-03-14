import numpy as np
import types


def default(val, d):
    if val is not None:
        return val
    return d() if isinstance(d, types.FunctionType) else d


class PLMS(object):
    def __init__(self, model,
                 ddim_steps,
                 batch_size,
                 shape,
                 eta=0.,
                 temperature=1.,
                 verbose=True,
                 log_every_t=100,
                 timesteps=1000,
                 linear_start=0.00085,
                 linear_end=0.0120,
                 dtype=np.float16) -> None:

        self.model = model
        self.batch_size = batch_size
        self.shape = shape
        self.temperature = temperature
        self.verbose = verbose
        self.log_every_t = log_every_t
        self.timesteps = timesteps
        self.linear_start = linear_start
        self.linear_end = linear_end
        self.dtype = dtype

        self.num_timesteps, self.betas, self.alphas_cumprod = self.register_schedule()
        self.ddim_sigmas, self.ddim_alphas, self.ddim_alphas_prev, self.ddim_timesteps = self.make_schedule(
            ddim_num_steps=ddim_steps, ddim_eta=eta, verbose=verbose)

    def register_schedule(self):

        betas = self.make_beta_schedule(
            self.timesteps, linear_start=self.linear_start, linear_end=self.linear_end)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        # alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        num_timesteps = int(timesteps)
        assert alphas_cumprod.shape[0] == timesteps, 'alphas have to be defined for each timestep'

        return num_timesteps, betas, alphas_cumprod

    def make_beta_schedule(self, n_timestep, linear_start=1e-4, linear_end=2e-2):
        betas = np.linspace(linear_start ** 0.5, linear_end **
                            0.5, n_timestep, dtype=self.dtype) ** 2

        return betas

    def make_schedule(self, ddim_num_steps,
                      ddim_discretize="uniform",
                      ddim_eta=0.,
                      verbose=True):
        if ddim_eta != 0:
            raise ValueError('ddim_eta must be 0 for PLMS')

        ddim_timesteps = self.make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  verbose=verbose)
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = self.make_ddim_sampling_parameters(ddim_timesteps,
                                                                                        ddim_eta, verbose)
        return ddim_sigmas, ddim_alphas, ddim_alphas_prev, ddim_timesteps

    def make_ddim_timesteps(self, ddim_discr_method, num_ddim_timesteps, verbose=True):
        if ddim_discr_method == 'uniform':
            c = self.timesteps // num_ddim_timesteps
            ddim_timesteps = np.asarray(list(range(0, self.timesteps, c)))
        elif ddim_discr_method == 'quad':
            ddim_timesteps = ((np.linspace(0, np.sqrt(
                self.timesteps * .8), num_ddim_timesteps)) ** 2).astype(int)
        else:
            raise NotImplementedError(
                f'There is no ddim discretization method called "{ddim_discr_method}"')

        # assert ddim_timesteps.shape[0] == num_ddim_timesteps
        # add one to get the final alpha values right (the ones from first scale to data during sampling)
        steps_out = ddim_timesteps + 1
        if verbose:
            print(f'Selected timesteps for ddim sampler: {steps_out}')
        return steps_out

    def make_ddim_sampling_parameters(self, ddim_timesteps, eta, verbose=True):
        # select alphas for computing the variance schedule
        alphas = self.alphas_cumprod[ddim_timesteps]
        alphas_prev = np.asarray(
            [self.alphas_cumprod[0]] + self.alphas_cumprod[ddim_timesteps[:-1]].tolist())

        # according the the formula provided in https://arxiv.org/abs/2010.02502
        sigmas = eta * np.sqrt((1 - alphas_prev) / (1 - alphas)
                               * (1 - alphas / alphas_prev))
        if verbose:
            print(
                f'Selected alphas for ddim sampler: a_t: {alphas}; a_(t-1): {alphas_prev}')
            print(f'For the chosen value of eta, which is {eta}, '
                  f'this results in the following sigma_t schedule for ddim sampler {sigmas}')
        return sigmas, alphas, alphas_prev

    def sample(self,
               batch_size,
               conditioning=None,
               mask=None,
               x0=None,
               x_T=None,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(
                        f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(
                        f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        C, H, W = self.shape
        size = (batch_size, C, H, W)
        samples, intermediates = self.plms_sampling(
            conditioning, size,
            x_T=x_T,
            mask=mask, x0=x0,
            unconditional_guidance_scale=unconditional_guidance_scale,
            unconditional_conditioning=unconditional_conditioning,
        )
        return samples, intermediates

    def plms_sampling(self, cond, shape,
                      x_T=None,
                      mask=None, x0=None,
                      unconditional_guidance_scale=1.,
                      unconditional_conditioning=None,
                      ):

        b = shape[0]
        if x_T is None:
            img = np.random.randn(*shape).astype(self.dtype)
            # img = np.zeros(shape).astype(self.dtype)
        else:
            img = x_T

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = np.flip(self.ddim_timesteps)
        total_steps = self.ddim_timesteps.shape[0]
        print(f"Running PLMS Sampling with {total_steps} timesteps")

        old_eps = []

        for i, step in enumerate(time_range):
            # print(f"i: {i} - step {step}")

            index = total_steps - i - 1
            ts = np.full((b,), step, dtype=np.int32)
            ts_next = np.full(
                (b,), time_range[min(i + 1, len(time_range) - 1)])

            if mask is not None:
                assert x0 is not None
                # TODO: deterministic forward pass?
                img_orig = self.q_sample(x0, ts)
                img = img_orig * mask + (1. - mask) * img

            outs = self.p_sample_plms(img, cond, ts, index=index,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning,
                                      old_eps=old_eps, t_next=ts_next)
            img, pred_x0, e_t = outs
            # print(img)
            old_eps.append(e_t)
            if len(old_eps) >= 4:
                old_eps.pop(0)

            if index % self.log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

        return img, intermediates

    def q_sample(self, x_start, t, noise=None, sqrt_alphas_cumprod=None, sqrt_one_minus_alphas_cumprod=None):
        noise = default(noise, lambda: np.random.randn(*x_start.shape))
        return (self.extract_into_tensor(sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                self.extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    def extract_into_tensor(self, a, t, x_shape):
        b, *_ = t.shape
        out = a.gather(-1, t)
        return out.reshape(b, *((1,) * (len(x_shape) - 1)))

    def p_sample_plms(self, x, c, t, index, repeat_noise=False,
                      unconditional_guidance_scale=1., unconditional_conditioning=None, old_eps=None, t_next=None):

        b = x.shape[0]

        def get_model_output(x, t):
            if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
                e_t = self.model.apply_model(x, t, c)
            else:
                if self.model.is_split_unet:
                    t_in = np.concatenate([t] * 2)
                else:
                    t_in = np.array(t)
                x_in = np.concatenate([x] * 2)
                # print('x_in:', x_in)
                c_in = np.concatenate([unconditional_conditioning, c])
                x_con = self.model.apply_model(
                    x_in, t_in, c_in)
                # print('x_con: ', x_con)
                e_t_uncond, e_t = np.split(x_con, 2)
                # print('e_t_uncond: ', e_t_uncond)
                # print('e_t', e_t)

                e_t = e_t_uncond + unconditional_guidance_scale * \
                    (e_t - e_t_uncond)
                # print('e_t: ', e_t)

            return e_t

        alphas = self.ddim_alphas
        alphas_prev = self.ddim_alphas_prev
        sqrt_one_minus_alphas = np.sqrt(1. - self.ddim_alphas)
        sigmas = self.ddim_sigmas

        def get_x_prev_and_pred_x0(e_t, index):
            # select parameters corresponding to the currently considered timestep
            a_t = np.full((b, 1, 1, 1), alphas[index], dtype=self.dtype)
            a_prev = np.full(
                (b, 1, 1, 1), alphas_prev[index], dtype=self.dtype)
            sigma_t = np.full((b, 1, 1, 1), sigmas[index], dtype=self.dtype)
            sqrt_one_minus_at = np.full(
                (b, 1, 1, 1), sqrt_one_minus_alphas[index], dtype=self.dtype)

            # current prediction for x_0
            pred_x0 = (x - sqrt_one_minus_at * e_t) / np.sqrt(a_t)
            # direction pointing to x_t
            dir_xt = np.sqrt((1. - a_prev - sigma_t**2)) * e_t
            noise = sigma_t * \
                self.noise_like(x.shape, repeat_noise) * self.temperature
            x_prev = np.sqrt(a_prev) * pred_x0 + dir_xt + noise
            # print('x_prev: ', x_prev)
            # print('pred_x0: ', pred_x0)
            return x_prev, pred_x0

        e_t = get_model_output(x, t)
        if len(old_eps) == 0:
            # Pseudo Improved Euler (2nd order)
            x_prev, pred_x0 = get_x_prev_and_pred_x0(e_t, index)
            e_t_next = get_model_output(x_prev, t_next)
            e_t_prime = (e_t + e_t_next) / 2
        elif len(old_eps) == 1:
            # 2nd order Pseudo Linear Multistep (Adams-Bashforth)
            e_t_prime = (3 * e_t - old_eps[-1]) / 2
        elif len(old_eps) == 2:
            # 3nd order Pseudo Linear Multistep (Adams-Bashforth)
            e_t_prime = (23 * e_t - 16 * old_eps[-1] + 5 * old_eps[-2]) / 12
        elif len(old_eps) >= 3:
            # 4nd order Pseudo Linear Multistep (Adams-Bashforth)
            e_t_prime = (55 * e_t - 59 *
                         old_eps[-1] + 37 * old_eps[-2] - 9 * old_eps[-3]) / 24

        x_prev, pred_x0 = get_x_prev_and_pred_x0(e_t_prime, index)

        return x_prev, pred_x0, e_t

    def noise_like(self, shape, repeat=False):
        def repeat_noise(): return np.repeat(np.random.randn(
            *(1, *shape[1:])), [shape[0], *((1,) * (len(shape) - 1))], axis=1).astype(self.dtype)

        def noise(): return np.random.randn(*shape).astype(self.dtype)
        return repeat_noise() if repeat else noise()
