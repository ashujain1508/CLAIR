<?php

    include 'logic.php';

?>

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">

    <!-- Bootstrap CSS -->
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.4.1/css/bootstrap.min.css">
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/font-awesome/4.7.0/css/font-awesome.min.css">

   <!-- Bootstrap JS -->
    <script src="https://code.jquery.com/jquery-3.4.1.slim.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/popper.js@1.16.0/dist/umd/popper.min.js"></script>
    <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.4.1/js/bootstrap.min.js"></script>

    <!-- Font Awesome -->
    <script src="https://kit.fontawesome.com/996973c893.js" crossorigin="anonymous"></script>

    <!-- Google Fonts -->
    <link href="https://fonts.googleapis.com/css2?family=Baloo+Thambi+2:wght@400;500;600;700;800&display=swap" rel="stylesheet">

    <!-- My Stylesheet -->
    <link rel="stylesheet" href="style.css">

    <title>Covid-19 </title>
</head>
<body style="background-color:rgba(241, 186, 186, 0.705);"> 

<div class="header-left" >
    <div class="menu" position="fixed">
      <ul >
        
        <li class="logo"><img src="/h1.jpg" alt="Computer Man" style="width:200px;height:100px;"></li>
        <li ><a href="" style="color:red;">       <i class="fa fa-home"></i>    Home</a></li>
        <li >  <a href="about.html" style="color:red;"><i class="fa fa-info" aria-hidden="true"></i>About</a>  </li>
        <li >  <a href="/final-files" style="color:red;"><i class="fa fa-shopping-cart" aria-hidden="true"></i>Shopping</a>  </li>
        <li><a href="contact.html" style="color:red;"><i class="fa fa-mobile" aria-hidden="true"></i>Contact</a></li>
        <li><a href="" onclick="javascript:event.target.port=5000" style="color:red;"><i class="fa fa-user-md" aria-hidden="true"></i><br>AI</a></li>
      </ul>

    </div>
</div>
<!--
<div class="header">
  <a href="#default" class="logo"><img src="https://www.animatedimages.org/data/media/118/animated-robot-image-0048.gif" alt="Computer Man" style="width:100px;height:100px;"></a>
  <div class="header-right">
    <a class="active" href="#home">Home</a>
    <a href="contact.html">Contact</a>
    <a href="about.html">About</a>
    <a href="/final-files">Admin-Shopping</a>
    <a href="" onclick="javascript:event.target.port=5000">AI Tools</a>
  </div>
</div>
-->


<div style="background-image: url('h2.jpg'); opacity: 0.95;">
<br>
<br>
<br>
<br>
<br>
<br>

 
<br>
<br>
    <div class="container my-5" style=" text-align: center; color:black; background-color:rgba(151, 23, 14, 0.514); border-radius:6px; border: 3px solid black;" >
        <h1 style=" font-size:50px;">Covid-19 Tracker</h1>
       
    
    <br>
    <br>
    
   <div  class="container my-5" style="background-color:rgba(151, 23, 14, 0.05); " >
        <div  class="row text-center">
            <div class="col-4 text-fail" style="color:rgb(0, 10, 44);">
                <h5>Confirmed</h5>
                <?php echo $total_confirmed;?>
            </div>
            <div class="col-4 text-fail" style="color:rgb(3, 70, 23);">
                <h5>Recovered</h5>
                <?php echo $total_recovered;?>
            </div>
            <div class="col-4 text-fail" style="color:rgb(68, 0, 0);">
                <h5>Deceased</h5>
                <?php echo $total_deaths;?>
            </div>
        </div>
        
    </div>
    </div>
    <div class="container bg-rgb(247, 180, 180) p-3 my-5 text-center" style="background-color:rgba(151, 23, 14, 0.514); border-radius:6px; border: 3px solid black;">
        <h5 style="color:white;">"Prevention is the Cure."</h5>
        <p style="color:white;">Stay Indoors Stay Safe.</p>
    </div>

    
<br>
<br>
<br>    
<br>
<br>
<br>    
<br>
<br>
<br>
</div>
    <div class="container-fluid" >
        <div class="table-responsive">
            <table class="table">
                <thead  >
                    <tr style="color:red; background-color:black;" >
                        <th scope="col">Countries</th>
                        <th scope="col">Confirmed</th>
                        <th scope="col">Recovered</th>
                        <th scope="col">Deceased</th>
                    </tr>
                </thead>
                <tbody>
                    <?php
                        foreach($data as $key => $value){
                            $increase = $value[$days_count]['confirmed'] - $value[$days_count_prev]['confirmed'];
                    ?>
                        <tr>
                            <th scope="row"><?php echo $key?></th>
                            <td>
                                <?php echo $value[$days_count]['confirmed'];?>
                                <?php if($increase != 0){ ?>
                                    <small class="text-danger pl-3"><i class="fas fa-arrow-up"></i><?php echo $increase;?></small>  
                                <?php } ?>    
                            </td>
                            <td><?php echo $value[$days_count]['recovered'];?></td>
                            <td><?php echo $value[$days_count]['deaths'];?></td>
                        </tr>
                    <?php }?>
                </tbody>
            </table>
        </div>
    </div>

    <footer class="footer mt-auto py-3 bg-light">
        <div class="container text-center">
            <span class="text-muted">Copyright &copy;2020, <a href="https://packetcode.in" target="_blank">Packetcode</a></span>
        </div>
    </footer>

 <script>
            window.watsonAssistantChatOptions = {
                integrationID: "9f6af214-7f74-4ce6-923e-40e29c078712", // The ID of this integration.
                region: "eu-gb", // The region your integration is hosted in.
                serviceInstanceID: "17db458f-d1f9-4dd1-9ef9-6a856eac728f", // The ID of your service instance.
                onLoad: function(instance) { instance.render(); }
              };
            setTimeout(function(){
              const t=document.createElement('script');
              t.src="https://web-chat.global.assistant.watson.appdomain.cloud/loadWatsonAssistantChat.js";
              document.head.appendChild(t);
            });
          </script>
          <script>
            window.watsonAssistantChatOptions = {
              // A UUID like '1d7e34d5-3952-4b86-90eb-7c7232b9b540' included in the embed code provided in Watson Assistant.
              integrationID: "9f6af214-7f74-4ce6-923e-40e29c078712",
              // Your assistants region e.g. 'us-south', 'us-east', 'jp-tok' 'au-syd', 'eu-gb', 'eu-de', etc.
              region: "eu-gb",
              // The callback function that is called after the widget instance has been created.
              onLoad: function(instance) {
                instance.updateCSSVariables({
                'BASE-font-family': '"Times New Roman", Times, serif',
                '$focus': '#000000',
                '$hover-primary': '#FFD700',
                '$interactive-01': '#800000'
      });
                instance.render();
              }
            };
            setTimeout(function(){const t=document.createElement('script');t.src='https://web-chat.global.assistant.watson.appdomain.cloud/loadWatsonAssistantChat.js';document.head.appendChild(t);});
          </script>

</body>
</html>