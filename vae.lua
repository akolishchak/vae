--
--
--  vae.lua
--  Created by Andrey Kolishchak on 2/14/16.
--
--
-- Variational auto-encoder, Guassian prior on latent space distribution
-- based on Auto-Encoding Variational Bayes, http://arxiv.org/abs/1312.6114
-- https://github.com/y0ast/VAE-Torch
--
--

require 'nn'
require 'Gaussian'
require 'GaussianCriterion'
require 'KLDGaussianCriterion'


local vae = {}

function vae.model_gaussian(opt)
  
  local encoder = nn.Sequential()
  encoder:add(nn.Linear(opt.input_dim, opt.hidden_layer_dim))
  encoder:add(nn.ELU())
  encoder:add(nn.ConcatTable()
                :add(nn.Linear(opt.hidden_layer_dim, opt.latent_dim)) -- mu(x)
                :add(nn.Linear(opt.hidden_layer_dim, opt.latent_dim)) -- log(sigma^2(x))
             )
  
  local decoder_net = nn.Sequential()
  decoder_net:add(nn.Linear(opt.latent_dim, opt.hidden_layer_dim))
  decoder_net:add(nn.ELU())
  decoder_net:add(nn.ConcatTable()
                    :add(nn.Linear(opt.hidden_layer_dim, opt.input_dim)) -- mu(z)
                    :add(nn.Linear(opt.hidden_layer_dim, opt.input_dim)) -- log(sigma^2(z))
                 )
  
  local decoder = nn.Sequential()
  decoder:add(nn.Gaussian()) -- z
  decoder:add(decoder_net)
  
  local model = nn.Sequential()
  model:add(encoder)
  model:add(nn.ConcatTable()
              :add(nn.Identity()) -- output for KL divergence Gaussian Criterion
              :add(decoder) -- output for Gaussian Criterion
           )
  
  return model, decoder_net
end

function vae.criterion_gaussian(opt)
  local criterion = nn.ParallelCriterion()
                      :add(nn.KLDGaussianCriterion(), 1)
                      :add(nn.GaussianCriterion(), 1)
  
  return criterion
end

function vae.model_bce(opt)
  
  local encoder = nn.Sequential()
  encoder:add(nn.Linear(opt.input_dim, opt.hidden_layer_dim))
  encoder:add(nn.Tanh())
  encoder:add(nn.ConcatTable()
                :add(nn.Linear(opt.hidden_layer_dim, opt.latent_dim)) -- mu(x)
                :add(nn.Linear(opt.hidden_layer_dim, opt.latent_dim)) -- log(sigma^2(x))
             )
  
  local decoder_net = nn.Sequential()
  decoder_net:add(nn.Linear(opt.latent_dim, opt.hidden_layer_dim))
  decoder_net:add(nn.Tanh())
  decoder_net:add(nn.Linear(opt.hidden_layer_dim, opt.input_dim)) -- y
  decoder_net:add(nn.Sigmoid())
  
  local decoder = nn.Sequential()
  decoder:add(nn.Gaussian()) -- z
  decoder:add(decoder_net)
  
  local model = nn.Sequential()
  model:add(encoder)
  model:add(nn.ConcatTable()
              :add(nn.Identity()) -- output for KL divergence Gaussian Criterion
              :add(decoder) -- output for BCE Criterion
           )
  
  return model, decoder_net
end

function vae.criterion_bce(opt)
  local criterion = nn.ParallelCriterion()
                      :add(nn.KLDGaussianCriterion(), 0.01)
                      :add(nn.BCECriterion(), 1)
  
  return criterion
end


return vae
