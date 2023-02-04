import torch
from torch.utils.data import TensorDataset
from card.architectures import RegressionNoiseEstimator
import math
from card.toysamplers import get_correct_mean

def PICP(model: RegressionNoiseEstimator, data: TensorDataset, device, lowQuantile = 0.025, highQuantile = 0.975):
    ## Needed variables
    picp = 0
    quants = torch.Tensor([lowQuantile, highQuantile])

    ## Calculating the Picp value
    for x,y in data:
        x_expand = x.expand([1000, 1]).to(device)
        yEst = model.infer(x_expand)

        yQuants = torch.quantile(yEst, quants)
        if y > yQuants[0] and y < yQuants[1]:
            picp += 1

    return picp/len(data)

def QICE(model: RegressionNoiseEstimator, data: TensorDataset, device, numBuckets = 10):
    ## Needed variables
    buckets = torch.zeros(numBuckets)
    quantiles = torch.arange(1, numBuckets) * 1/numBuckets

    ## Calculating the Qice value
    for x,y in data:
        x_expand = x.expand([1000, 1]).to(device)
        yEst = model.infer(x_expand)
        yQuants = torch.quantile(yEst, quantiles)
        for i in range(numBuckets - 1):
            if y < yQuants[i]:
                buckets[i] += 1
                break
        else:
            buckets[numBuckets-1] +=1

    buckets /= len(data)
    buckets -= 1/numBuckets
    buckets = torch.sum(torch.abs(buckets)).item()
    
    return buckets/numBuckets

def combined_metrics(model: RegressionNoiseEstimator,
                     data: TensorDataset,
                     device,
                     mean_y: torch.Tensor,
                     std_y: torch.Tensor,
                     picp_width: float = 0.95,
                     num_buckets_qice: int = 10,
                     dataset_name: str = None):
    # Put means and std devs on gpu
    mean_y = mean_y.to(device)
    std_y = std_y.to(device)
    
    # Needed variables
    target_size = data.dataset.tensors[1].shape[1]
    if target_size == 1:
        low_quantile_picp = (1.0 - picp_width) /2 
    else:
        low_quantile_picp = (1.0 - torch.sqrt(torch.Tensor((picp_width,)))) / 2 
    high_quantile_picp = 1.0 - low_quantile_picp
    picp = 0
    quantiles_picp = torch.Tensor([low_quantile_picp, high_quantile_picp]).to(device)
    buckets = torch.zeros(num_buckets_qice).to(device)
    quantiles_qice = (torch.arange(1, num_buckets_qice) * 1/num_buckets_qice).to(device)
    rmse = 0
    nll = 0

    for sample_idx, batch in enumerate(data):
        x, y = batch
        x, y = x.to(device), y.to(device)

        # Sample 1000 model predictions to evaluate metrics on.
        x_expanded = x.expand([1000, *(x.shape)])
        y_preds = model.infer(x_expanded)

        # Calculate PICP
        y_quants_picp = torch.quantile(y_preds, quantiles_picp).to(device)
        if torch.all(torch.gt(y, y_quants_picp[0])) and torch.all(torch.lt(y , y_quants_picp[1])):
            picp += 1

        # Calculate QICE
        y_quants_qice = torch.quantile(y_preds, quantiles_qice).to(device)
        for j in range(target_size):
            for i in range(num_buckets_qice - 1):
                if torch.lt(y,y_quants_qice[i])[j]:
                    buckets[i] += 1
                    break
            else:
                buckets[num_buckets_qice-1] +=1

        # Calculate RMSE
        if len(y_preds.shape) > 2:
            y_preds = y_preds.squeeze(-1)
            y = y.reshape((1, 2))
            std_y = std_y.reshape((1, 2))
            mean_y = mean_y.reshape((1, 2))
        y_preds = y_preds * std_y + mean_y
        y_target = y * std_y + mean_y
        if dataset_name in ('Naval', 'Kin8nm'):
            y_preds *= 100
            y_target *= 100

        mean_y_estimates = torch.mean(y_preds, dim=0)
        std_y_estimates = torch.std(y_preds, dim=0)
        if len(mean_y_estimates.shape) == 0:
            mean_y_estimates = mean_y_estimates.reshape((1,))
            std_y_estimates = std_y_estimates.reshape((1,))
        sigma_squared = 2 * torch.matmul(std_y_estimates, std_y_estimates)

        if len(y_target.shape) > 1:
            y_target = y_target.reshape((2,))
        diff = mean_y_estimates - y_target

        diff = torch.matmul(diff, diff)
        rmse += diff

        # Calculate NLL
        probDF = - diff / sigma_squared
        probDF = torch.exp(probDF)
        probDF /= torch.sqrt(math.pi * sigma_squared)
        nll -= torch.log(probDF)


        ## Print for sanity checks and bugfixing
        if sample_idx % 10 == 0:
            print(f'Datapoint = {sample_idx:>3} / {len(data)}, Picp = {picp}, Buckets = {buckets}')

    # Final Calculations Qice
    buckets /= (len(data) * target_size)
    buckets -= 1/num_buckets_qice
    buckets = torch.sum(torch.abs(buckets)).item()

    return picp/len(data), buckets/num_buckets_qice, float(torch.sqrt(rmse/len(data))), float((nll/len(data)))

def toy_metrics(model: RegressionNoiseEstimator, data: TensorDataset, device, experiment_index, mean_y, std_y, lowQuantilePicp = 0.025, highQuantilePicp = 0.975, numBucketsQice = 10):
    ## Needed variables
    counter = 0
    picp = 0
    quantilesPicp = torch.Tensor([lowQuantilePicp, highQuantilePicp]).to(device)
    buckets = torch.zeros(numBucketsQice).to(device)
    quantilesQice = (torch.arange(1, numBucketsQice) * 1/numBucketsQice).to(device)
    rmse = 0
    nll = 0


    for x,y in data:
        x, y = x.to(device), y.to(device)
        counter += 1
        x_expand = x.expand([1000, *(x.shape)])
        yEst = model.infer(x_expand)

        ##Calculating Picp
        yQuantsPicp = torch.quantile(yEst, quantilesPicp).to(device)
        if y > yQuantsPicp[0] and y < yQuantsPicp[1]:
            picp += 1

        #Calculating Qice
        yQuantsQice = torch.quantile(yEst, quantilesQice).to(device)
        for i in range(numBucketsQice - 1):
            if y < yQuantsQice[i]:
                buckets[i] += 1
                break
        else:
            buckets[numBucketsQice-1] +=1

        #Calculating RMSE
        yEst = yEst * std_y + mean_y
        mean_yEst = torch.mean(yEst)

        yTrue = get_correct_mean(x, experiment_index)
        yTrue = yTrue * std_y + mean_y

        diff = mean_yEst - yTrue
        diff = diff ** 2
        rmse += diff


        ## Print for sanity checks and bugfixing
        if counter % 10 == 0:
            print(f'Datapoint = {counter:>3} / {len(data)}, Picp = {picp}, Buckets = {buckets}')

    # Final Calculations Qice
    buckets /= len(data)
    buckets -= 1/numBucketsQice
    buckets = torch.sum(torch.abs(buckets)).item()

    return picp/len(data), buckets/numBucketsQice, float(torch.sqrt(rmse/len(data))[0])