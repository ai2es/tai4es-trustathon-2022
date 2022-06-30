import tqdm
import copy 
import gc 
import sklearn.metrics
import numpy as np


class ImagePermI:
    """ class to run permutation importance with """ 
    def __init__(self,images,model,labels, subsample_size=1.0, n_permute=10, 
                 seed=42,verbose=True):
    
        rs = np.random.RandomState(123)
        subsample_size = int(subsample_size*len(images)) if subsample_size <= 1.0 else subsample_size
        if subsample_size == len(images):
            inds = np.arange(len(images))
        else:
            inds = rs.choice(len(images), size=subsample_size)
            
        images = images[inds,:,:,:]
        labels = labels[inds]
        
        self.orig_images = images 
        self.model = model
        self.seed = seed
        self.labels = labels
        self.all_shuffled = None
        self.verbose = verbose
        self.n_permute = n_permute
        
    def shuffle_images(self,in_images,channels,shuff_sampledim=True):
        """ 
        expects [n_samples,nx,ny,n_channel]
        
        params:
        
        channels: arraylike, will shuffle all channels provided
        
        """
        #make copy of input images 
        images_new = copy.deepcopy(in_images)
        
        #shuffle over all channels given 
        for channel in channels:
            #reshape so we can shuffle all the pixels in an image first 
            img_tmp = np.reshape(images_new[:,:,:,channel],[images_new.shape[0],images_new.shape[1]*images_new.shape[2]])
            img_shuffled = self.genor.permutation(img_tmp, axis=1)
            #reshape back 
            img_shuffled = np.reshape(img_shuffled,[img_shuffled.shape[0],images_new.shape[1],images_new.shape[2]])
            
            #now shuffle n_sample dim (i.e., shuffle maps)
            if shuff_sampledim:
                idx_sample = np.random.randint(0,img_shuffled.shape[0],size=img_shuffled.shape[0])
                img_shuffled = img_shuffled[idx_sample]
                
            images_new[:,:,:,channel] = img_shuffled
            
        del img_tmp,img_shuffled 
        
        #return shuffled images
        return images_new
        
    def get_score(self,in_images, metric='auc'):
        self.preds = np.ravel(self.model.predict(in_images))
        if metric=='auc':
            score = sklearn.metrics.roc_auc_score(self.labels,self.preds)
        elif metric == 'mse':
            score = sklearn.metrics.mean_squared_error(self.labels,self.preds)
        return score 
            
    def single_pass(self, metric, leave_alone=None,direction='backward'):
        #initalize random gen so the shuffles are the same for each single pass 
        #self.genor = np.random.default_rng(self.seed)
        
        self.genor = np.random.default_rng(self.seed)
        
        #to save compute see if all_shuffled array exists, if not shuffle all data for use later
        if self.all_shuffled is None:
            if self.verbose:
                print('Shuffling Images, please wait...')
            self.all_shuffled = self.shuffle_images(self.orig_images,channels=np.arange(0,self.orig_images.shape[-1]))
            if self.verbose:
                print('done')
        else:
            pass
            
            
        #all idx 
        idx_to_consider = np.arange(0,self.orig_images.shape[-1])
        
        #initialize score array
        self.scores = np.zeros(self.orig_images.shape[-1])
            
        #make copy of images
        if direction == 'backward':
            #if backward source original images 
            image_array = copy.deepcopy(self.orig_images)
        elif direction == 'forward':
            #if foreward source shuffled images 
            image_array = copy.deepcopy(self.all_shuffled)
            
        #check to see if any channels should be left alone 
        if leave_alone is None:
            #if we are in the first call, do nothing 
            pass
        else:
            #if not step 0 of multi-pass, insert unchanged array 
            for i in leave_alone:
                if direction == 'backward':
                    #if backward, fill with shuffled 
                    image_array[:,:,:,i] = copy.deepcopy(self.all_shuffled[:,:,:,i])
                elif direction == 'forward':
                    #if foreward, fill with orignial images 
                    image_array[:,:,:,i] = copy.deepcopy(self.orig_images[:,:,:,i])

            #get idx needed to run experiment on 
            idx_to_consider = np.setxor1d(idx_to_consider, leave_alone)

        #get starting score of the pass 
        self.start_score = self.get_score(image_array, metric=metric)

        #loop over all channels you want to consider
        for i in tqdm.tqdm(idx_to_consider):

            #make copy of starting array 
            image_tmp = copy.deepcopy(image_array)

            #replace channel under consideration
            if direction == 'backward':
                image_tmp[:,:,:,i] = self.all_shuffled[:,:,:,i]
            elif direction == 'forward':
                image_tmp[:,:,:,i] = self.orig_images[:,:,:,i]

            #compute score with altered array 
            self.scores[i] = self.get_score(image_tmp, metric=metric)

        #free up RAM, del big temporary variables 
        del image_tmp, image_array
                
    def multi_pass(self,direction='backward'):
        
        print("CODE IS NOT FINISHED. PLEASE CHECK BACK LATER")
        return 
        winners = np.array([],dtype=int)
        self.all_scores = np.zeros(self.orig_images.shape[-1])
        #need to call this n many times 
        for n in np.arange(0,self.orig_images.shape[-1]):
            if self.verbose:
                print('single pass {}:'.format(n))
            
            #if the first call do dont give any idx
            if n == 0:
                self.single_pass(direction=direction)
                #save original score 
                self.og_score = copy.deepcopy(self.start_score)
            else:
                #send back to single pass, dont change the winners 
                self.single_pass(direction=direction,leave_alone=winners)
            

                
            if self.verbose:    
                print(self.start_score,self.scores)
            
            #determine difference from starting score 
            differ = np.abs(self.start_score - self.scores)
            
            #fill in winner scores with 0 so they dont get picked again 
            for w in winners:
                differ[w] = 0
                
            #determine top influencer
            idx_winner = np.argmax(differ)
            #save the max score 
            self.all_scores[idx_winner] = self.scores[idx_winner]
            
            #save the winner id
            winners = np.append(winners,idx_winner)
            
            #run garbage collect to clean things up in memory 
            gc.collect()
        
        #get final score
        if direction == 'backward':
            self.backward = {}
            self.backward['end_score'] = self.get_score(self.all_shuffled)
            self.backward['start_score'] = self.og_score
            self.backward['id_order'] = winners 
            
        elif direction == 'forward':
            self.forward = {}
            self.forward['end_score'] = self.get_score(self.orig_images)
            self.forward['start_score'] = self.og_score
            self.forward['id_order'] = winners
