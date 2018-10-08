mydata <- read.csv("hw1q6_data.csv")
classNumbers<-mydata[["Class"]]
glucose <-mydata[["Glucose"]]
bloodpressure<- mydata[["BloodPressure"]]
skinthickness <- mydata[["SkinThickness"]]
bmi<-mydata[["BMI"]]
dpf <-mydata[["DiabetesPedigreeFunction"]]
age <-mydata [["Age"]]


# Question a
#Number of diabetic and Non-diabetic patients -Question1
diabeticNum=length(which(classNumbers==1))
nondiabeticNum=length(which(classNumbers==0))

# Question b
#missing rate for Glucose,BloodPressure,SkinThickness,BMI,DiabetesPedigreeFunction,Age
glucoseMissingRate= (length(which(glucose==0))/length(glucose))*100
bpMissingRate =(length(which(bloodpressure==0))/length(bloodpressure))*100
stMissingRate=(length(which(skinthickness==0))/length(skinthickness))*100
bmiMissingRate =(length(which(bmi==0))/length(bmi))*100
dpfMissingRate = (length(which(dpf==0))/length(dpf))*100
ageMissingRate=(length(which(age==0))/length(age))*100

#Dataset with no missing values
mydata_deleted <- read.csv("hw1q6_data_1.csv")
classNumbers_deleted<-mydata_deleted[["Class"]]
glucose_deleted <-mydata_deleted[["Glucose"]]
bloodpressure_deleted<- mydata_deleted[["BloodPressure"]]
skinthickness_deleted <- mydata_deleted[["SkinThickness"]]
bmi_deleted<-mydata_deleted[["BMI"]]
dpf_deleted <-mydata_deleted[["DiabetesPedigreeFunction"]]
age_deleted <-mydata_deleted [["Age"]]


#Question d
#Number of diabetic and Non-diabetic patients -Question1
diabeticNum_deleted=length(which(classNumbers_deleted==1))
nondiabeticNum_deleted=length(which(classNumbers_deleted==0))

#Question e
#Glucose
g_mean = mean(glucose_deleted)
g_median= median(glucose_deleted)
g_std= sd(glucose_deleted)
g_range=range(glucose_deleted)
g_quantile=quantile(glucose_deleted, c(.25, .50,.75))

#Blood Pressure
bp_mean = mean(bloodpressure_deleted)
bp_median= median(bloodpressure_deleted)
bp_std= sd(bloodpressure_deleted)
bp_range=range(bloodpressure_deleted)
bp_quantile=quantile(bloodpressure_deleted, c(.25, .50,.75))

#Skin Thickness
st_mean = mean(skinthickness_deleted)
st_median= median(skinthickness_deleted)
st_std= sd(skinthickness_deleted)
st_range=range(skinthickness_deleted)
st_quantile=quantile(skinthickness_deleted, c(.25, .50,.75))

#Skin Thickness
bmi_mean = mean(bmi_deleted)
bmi_median= median(bmi_deleted)
bmi_std= sd(bmi_deleted)
bmi_range=range(bmi_deleted)
bmi_quantile=quantile(bmi_deleted, c(.25, .50,.75))

#DiabetesPedigreeFunction
dpf_mean = mean(dpf_deleted)
dpf_median= median(dpf_deleted)
dpf_std= sd(dpf_deleted)
dpf_range=range(dpf_deleted)
dpf_quantile=quantile(dpf_deleted, c(.25, .50,.75))

#Age
age_mean = mean(age_deleted)
age_median= median(age_deleted)
age_std= sd(age_deleted)
age_range=range(age_deleted)
age_quantile=quantile(age_deleted, c(.25, .50,.75))

#Question f
#Blood Pressure histogram

bp_plot=hist(bloodpressure_deleted,breaks=seq(min(bloodpressure_deleted),max(bloodpressure_deleted),l=11), 
             main="Histogram",col="orange",xlab="Blood Pressure",ylab="Frequency")
#Diabetes Pedigree function histogram

dpf_plot=hist(dpf_deleted,breaks=seq(min(dpf_deleted),max(dpf_deleted),l=11), 
             main="Histogram",col="orange", xlab="Diabetes Pedigree function",ylab="Frequency")


# Question g
#Blood Pressure Quantile Quantile plot
qqplot_bp=qqnorm(bloodpressure_deleted,col = "steelblue")

#Diabetes Pedigree function Quantile Quantile plot
qqplot_dpf=qqnorm(dpf_deleted, col = "orange")
  
