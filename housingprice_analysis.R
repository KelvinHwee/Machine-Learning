####################################################################################################
###   load in the required libraries
####################################################################################################
# ensure that the required packages are installed
# install.packages("dplyr")
# install.packages("ggplot2")
# install.packages("httr")
# install.packages("stringr")

library("dplyr")
library("ggplot2")
library("httr")
library("stringr")
require("httr")


####################################################################################################
###   read in the CSV files (to be stored directly in the working directory, and not in any folder)
####################################################################################################

###   we read files in different batches
df_resale1 <- read.csv("~/resale-flat-prices-based-on-approval-date-1990-1999.csv", stringsAsFactors = FALSE)
df_resale2 <- read.csv("~/resale-flat-prices-based-on-approval-date-2000-feb-2012.csv", stringsAsFactors = FALSE)
df_resale3 <- read.csv("~/resale-flat-prices-based-on-registration-date-from-mar-2012-to-dec-2014.csv", stringsAsFactors = FALSE)
df_resale4 <- read.csv("~/resale-flat-prices-based-on-registration-date-from-jan-2015-to-dec-2016.csv", stringsAsFactors = FALSE)
df_resale5 <- read.csv("~/resale-flat-prices-based-on-registration-date-from-jan-2017-onwards.csv", stringsAsFactors = FALSE)


####################################################################################################
###   we do some basic clean-up
####################################################################################################

###   there is extra column in "df_resale4" and "df_resale5", we drop the additional "remaining_lease" column
###   since this info is not available in the earlier files
df_resale4 <- select(df_resale4, -c("remaining_lease"))
df_resale5 <- select(df_resale5, -c("remaining_lease"))

###   combine the dataframes into one dataframe
df_resale <- rbind(df_resale1, df_resale2, df_resale3, df_resale4, df_resale5)

###   we further drop columns that are not critical for our investigation into the r/s between price 
###   against factors like location and structural characteristics (e.g. size)
df_resale <- select(df_resale, -c("lease_commence_date"))

###   engineer a new "year" column for EDA purposes
df_resale$year <- str_sub(df_resale$month, 1, 4)
dim(df_resale)

###   take a look at an extract of the combined dataframe
head(df_resale); tail(df_resale)

###   there are repeated info, e.g. rows 1 to 4 are all the same except for "storey_range" and "resale_price"
###   we consider calculating an average resale_price, then thereafter, collapse these columns, else too many columns
###   also note that we are looking at just 2019's numbers (afterall, some flats may no longer exist)
###   the last "mutate" creates a column that combines blk number and street name
df <- df_resale %>%
        group_by(year, town, flat_type, block, street_name, floor_area_sqm, flat_model) %>%
            summarise(avg_resale_price = mean(resale_price)) %>%
                filter(year == "2019") %>%
                    mutate(blk_street_name = str_replace_all(paste(block, street_name), " ", "%20"))

head(df)
dim(df)

####################################################################################################
###   we want to engineer latitude and longitude info into the dataframe, 
###   we need API access from Onemap.sg
####################################################################################################

###   we get the access token from onemap 
head_url <- "https://developers.onemap.sg"
post_url <- paste0(head_url, "/privateapi/auth/post/getToken")
body <- list('email' = 'hwee.yewrong.kelvin@gmail.com', 'password' = 'testing123') # for purpose of assignment, i used this approach
res <- POST(post_url, body = body, verbose())
content(res)

###   we create a function to perform a GET request
get_request <- function(addr, index){
    
    head_url <- "https://developers.onemap.sg"
    get_url <- paste0(head_url, "/commonapi/search", "?searchVal=", addr, "&returnGeom=Y&getAddrDetails=N&pageNum=1")
    get_res <- GET(get_url, verbose())
    
    if(content(get_res)[1] > 0){    lat <- content(get_res)$'results'[[1]]$'LATITUDE'
                                    long <- content(get_res)$'results'[[1]]$'LONGITUDE'
                                    lat_long <- c(lat, long)
    } else{   lat <- 'not found'
              long <- 'not found'
              lat_long <- c(lat, long)}
    
    result <- ifelse(index == "lat", lat_long[1], lat_long[2])
    return(result)
}


df_selected <- df

###   creating columns called "latitude" and "longitude"
# for(i in c(1:length(df_selected$blk_street_name)))
#     df_selected$latitude[i] <- get_request(df_selected$blk_street_name[i], "lat")
# 
# for(i in c(1:length(df_selected$blk_street_name)))
#     df_selected$longitude[i] <- get_request(df_selected$blk_street_name[i], "long")
# 
# str(df_selected)
# head(df_selected)    
# 

# df_selected[df_selected$latitude == "not found",]  # we found some rows with missing lat long
# indexes <- df_selected[df_selected$latitude == "not found", 1] # we collect the corresponding index numbers
# 
# 
# # we also separately used Singpost's website to retrieve the postal code, and record them here in order
# code_list = c("320007", "320014", "328023", "321020", 
#               "320003", "320005", "320006", "320008",
#               "320001", "320010", "320011", "320014",
#               "320020", "320023", "321004", "320005",
#               "320005", "320006", "320006", "320008",
#               "320009", "320013", "328023", "321021")
# 
# df_selected <- read.csv("~/exported_resale_data.csv", stringsAsFactors = FALSE)
# 
# for(i in c(1:length(indexes)))
#     df_selected$latitude[indexes[i]] <- get_request(code_list[i], "lat")
# 
# for(i in c(1:length(indexes)))
#     df_selected$longitude[indexes[i]] <- get_request(code_list[i], "long")
# 
#     
# df_selected$latitude[indexes]
# df_selected$longitude[indexes]
# 
# write.csv(df_selected, paste0(getwd(), "\\exported_resale_data.csv"))


####################################################################################################
###   we do some exploratory data analysis
####################################################################################################

# we start by reading in the cleaned up data that was created in the earlier step
df_selected <- read.csv("~/exported_resale_data.csv", stringsAsFactors = FALSE)
df_selected$latitude <- as.numeric(df_selected$latitude)
df_selected$longitude <- as.numeric(df_selected$longitude)
df_selected$log_avg_price <- log(df_selected$avg_resale_price)
head(df_selected)

sapply(df_selected, function(x) sum(is.na(x))) # no null values

###   ranking of prices based on town (flat type as colour)
qplot(town, avg_resale_price, data = df_selected, 
      color = flat_type, facets = .~flat_type) +
      geom_smooth(method = "lm") + 
      theme(axis.text.x = element_text(angle = 90)) +
      theme(axis.text.x = element_text(size = 5))

###   ranking of prices based on town (floor area as colour)
df_selected$cut_floor_area_sqm <- cut(df_selected$floor_area_sqm, 6)
qplot(town, avg_resale_price, data = df_selected, 
      color = cut_floor_area_sqm, facets = .~cut_floor_area_sqm) +
      geom_smooth(method = "lm") + 
      theme(axis.text.x = element_text(angle = 90)) + 
      theme(axis.text.x = element_text(size = 5))

###   ranking of prices based on town (flat model as colour)
qplot(town, avg_resale_price, data = df_selected, 
      color = flat_model, facets = .~flat_model) +
      geom_smooth(method = "lm") + 
      theme(axis.text.x = element_text(angle = 90)) +
      theme(axis.text.x = element_text(size = 5))

###   ranking of prices based on town (flat model as colour)
qplot(town, avg_resale_price, data = df_selected[df_selected$flat_model == c("Type S1", "Type S2"),], 
      fill = flat_model, facets = .~flat_model) +
      geom_smooth(method = "lm") + geom_boxplot()

###   we do a cut on the average prices and plot onto the map
df_selected$cut_avg_resale_price <- cut(df_selected$avg_resale_price, 5)
ggplot(data = df_selected, aes(x=latitude, y=longitude)) + 
    geom_point(aes(alpha = 0.6, color = cut_avg_resale_price)) +
    scale_colour_manual(values=c("red", "pink", "dark blue", "light green", "green"))
    
###   we use the cut average prices and do a bar plot
ggplot(data = df_selected, aes(x=reorder(town, -avg_resale_price), fill = cut_avg_resale_price)) +
    geom_bar(position = "dodge", color = "black") + 
    theme(axis.text.x = element_text(angle = 90)) +
    scale_fill_manual(values=c("red", "pink", "dark blue", "light green", "green"))

###   we do a boxplot of the town and the average resale prices
ggplot(data = df_selected, aes(x=reorder(town, -avg_resale_price), y=avg_resale_price)) + 
    geom_boxplot() + geom_point(aes(color = cut_avg_resale_price)) + 
    theme(axis.text.x = element_text(angle = 90))



