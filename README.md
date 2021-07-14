# Genetic Driver 
(for TORCS)

Genetic Algorithm implementation for the TORCS SCR Championship software, using Python 3 and Keras.

## `My_Driver`
* Has all the functionalities of the driving logic and execute the commands for the car actuation.
	
## `Run`
* Main entrance of the program, has the structure of the Genetic Algorithm.

## `Genetics`
* Python file with all the implementations of the Genetic Algorithm functions.
	
## `Modelo`
* Implementation of the Neural Network architecture with Keras and the management of it's weights.
	

## Fork of torcs-client
-------------------------------------------
	- Python client for TORCS with network plugin for the 2012 SCRC

	This is a copy of the reimplementation in Python 3 by @moltob of the original SCRC TORCS client pySrcrcClient from @lanquarden. It is used to teach ideas of computational intelligence. The file `my_driver.py` contains a shell to start writing your own driver.

	-> `Client`

	* top level class
	* handles _all_ aspects of networking (connection management, encoding)
	* decodes class `State` from message from server, `state = self.decode(msg)`
	* encodes class `Command` for message to server, `msg = self.encode(command)`
	* internal state connection properties only and driver instance
	* use `Client(driver=your_driver, <other options>)` to use your own driver

	-> `Driver`

	* encapsulates driving logic only
	* main entry point: `drive(state: State) -> Command`

	-> `State`

	* represents the incoming car state

	-> `Command`

	* holds the outgoing driving command



