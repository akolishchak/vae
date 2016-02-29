--
--
--  main.lua
--  Created by Andrey Kolishchak on 2/14/16.
--
--
-- Variational auto-encoder
-- based on Auto-Encoding Variational Bayes, http://arxiv.org/abs/1312.6114
-- https://github.com/y0ast/VAE-Torch
--
--
require 'nn'
require 'optim'
require 'image'
require 'gnuplot'

require 'data.dataset'
local vae = require 'vae'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Variational auto-encoder on MNIST')
cmd:text()
cmd:text('Options')
cmd:option('-gpu',2,'0 - cpu, 1 - cunn, 2 - cudnn')
cmd:option('-latent_dim',2,'latent space dimension')
cmd:option('-hidden_layer_dim',500,'hidden layer dimension')
cmd:option('-learning_rate',1e-4,'learning rate')
cmd:option('-batch_size',100,'batch size')
cmd:option('-max_epoch',100,'number of passes through the training data')
cmd:option('-output_path','samples3','path for output images')

opt = cmd:parse(arg)

if opt.gpu > 0 then
  require 'cunn'
  if opt.gpu == 2 then
    require 'cudnn'
  end
end

--
-- load data
--
local dataset = load_mnist(opt)
dataset.train_x = dataset.train_x:view(dataset.train_x:size(1), -1)
dataset.train_x = dataset.train_x:ge(0) -- nomalize to the interval [0,1] for BCE criterion
--
opt.input_dim = dataset.train_x:size(2)

--
-- build model
--
torch.manualSeed(0)
local model, decoder_net = vae.model_bce(opt)
local criterion = vae.criterion_bce(opt)

if opt.gpu > 0 then
  model:cuda()
  criterion:cuda()
  
  if opt.gpu == 2 then
    cudnn.convert(model, cudnn)
    cudnn.convert(criterion, cudnn)
    cudnn.benchmark = true
  end
end

print(model)
print(criterion)

params, grad_params = model:getParameters()

--
-- optimize
--
local iterations = opt.max_epoch*dataset.train_x:size(1)/opt.batch_size
local batch_start = 1

function feval(x)
  if x ~= params then
    params:copy(x)
  end
  grad_params:zero()

  -- load batch
  local input = dataset.train_x[{{batch_start, batch_start+opt.batch_size-1},{}}]
  local target = {nil, input}
  
  -- forward pass
  local output = model:forward(input)
  local loss = criterion:forward(output, target)
  
  -- backward pass
  local dloss_doutput = criterion:backward(output, target)
  model:backward(input, dloss_doutput)
  
  return loss, grad_params
end

local optim_state = {learningRate = opt.learning_rate}

local losses = torch.Tensor(iterations)

model:training()

for it = 1,iterations do
  
    local _, loss = optim.adam(feval, params, optim_state)

    if it % 100 == 0 then
      print(string.format("batch = %d, loss = %.12f, grad/param norm = %.6f/%.6f", it, loss[1], grad_params:norm(), params:norm()))
    end
    
    if loss[1] < 1 then
      losses[it] = loss[1]
    else
      losses[it] = 0
    end
    
  
    batch_start = batch_start + opt.batch_size
    if batch_start > dataset.train_x:size(1) then
      batch_start = 1
    end 
    
end

model:evaluate()

require 'Gaussian'
local output = model:forward(dataset.train_x)
output = nn.Gaussian():cuda():forward(output[1])
print(output:mean(), output:std())
--print(output[{{1,200}, {}}])

--
-- take samples
--
paths.mkdir(opt.output_path)

if opt.latent_dim == 2 then
  -- linspace for 2 dimensions
  local width = torch.sqrt(dataset.train_x:size(2))
  local manifold = torch.Tensor(21*width, 21*width)
  local column = 0
  for i = -1,1,0.1 do
    local input = torch.Tensor(21, 2)
    if opt.gpu > 0 then input = input:cuda() end
    local row = 1
    for j = 1,-1,-0.1 do
      input[row][1] = i
      input[row][2] = j
      row = row + 1
    end
    manifold[{{},{column*width + 1, column*width+width}}]:copy(decoder_net:forward(input):ge(0.5):view(-1, width))
    image.save(opt.output_path..'/manifold.jpg', manifold)
    column = column+1
  end
  
else
  -- random samples for more dimensions
  local input = torch.Tensor(100, opt.latent_dim):normal(0,1)
  if opt.gpu > 0 then input = input:cuda() end
  local samples = decoder_net:forward(input)
  
  for i = 1,100 do
    image.save(opt.output_path..'/'..tostring(i)..'.jpg', samples[i]:view(28,28))
  end
end
