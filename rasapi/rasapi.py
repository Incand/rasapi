# This file is part of rasapi.

# rasapi is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# rasapi is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with rasapi.  If not, see <https://www.gnu.org/licenses/>.

# Copyright 2019, Armin Schaare <armin-scha@hotmail.de>

'''Module containing RASA API bindings.'''

from typing import NamedTuple, Union, Iterable, Optional, Dict, Text

import requests
import os
from functools import wraps

import logging
logger = logging.getLogger(__name__)


class Event(NamedTuple):
    '''Represents Rasa events which can be added/modified on a conversation
    tracker.

    Fields:
    event -- Name of the event
    timestamp -- When this event occured
    '''
    event: str
    timestamp: Optional[int] = None

    def asdict(self) -> Dict:
        '''Return the event as an API compatible dictionary.'''
        ret = {'event': self.event}
        if self.timestamp is not None:
            ret['timestamp'] = self.timestamp
        logger.debug(f'Event as dictionary: {ret}')
        return ret


def log_args_ret(level=logging.INFO):
    def deco(f):
        @wraps(f)
        def _wrapper(self, *args, **kwargs):
            global logger
            args_str_list = [repr(arg) for arg in args]
            kwargs_str_list = [f'{k}={repr(v)}' for k, v in kwargs.items()]
            param_string = ', '.join(args_str_list + kwargs_str_list)
            logger.log(level, f'{f.__qualname__}({param_string})')
            tmp_logger_name = logger.name
            logger = logger.getChild(f'{f.__qualname__}')
            ret = f(self, *args, **kwargs)
            logger = logging.getLogger(tmp_logger_name)
            logger.log(level, f' -> {ret}')
            return ret
        return _wrapper
    return deco


class RasaPI:
    '''Class for Rasa API communication.'''
    def __init__(self, url: str, token: Optional[str] = None) -> None:
        logger.info('Setting up RasaPI object...')
        self.url = url.strip('/')
        logger.debug(f'Using URL: {self.url}')
        if token is None:
            logger.debug('Token Argument is None. '
                         'Checking envvars for "RASA_TOKEN"...')
            token = os.environ.get('RASA_TOKEN', None)
            if token is None:
                logger.debug('"RASA_TOKEN" not found in envvars.')
        self.token = token
        logger.debug(f'Using token: {self.token}')

    def _request(self, method: str, path: str, **kwargs) -> requests.request:
        url = self.url + path
        req_args = {'method': method, 'url': url}
        token = self.token
        if token:
            kwargs.setdefault('params', {'token': token})
        req_args.update(kwargs)
        logger.debug(f'Request Args: {req_args}')
        resp = requests.request(**req_args)
        logger.debug(f'Response: {resp.text}')
        return resp

    def _get(self, path: str, **kwargs) -> requests.request:
        return self._request('GET', path, **kwargs)

    def _post(self, path: str, **kwargs) -> requests.request:
        return self._request('POST', path, **kwargs)

    def _put(self, path: str, **kwargs) -> requests.request:
        return self._request('PUT', path, **kwargs)

    def _delete(self, path: str, **kwargs) -> requests.request:
        return self._request('DELETE', path, **kwargs)

    @property
    @log_args_ret()
    def is_healthy(self) -> bool:
        '''Returns True if server is up and reachable, False otherwise.'''
        return self._get('/').ok

    @property
    @log_args_ret()
    def version(self) -> str:
        '''Get the version of the running Rasa server.'''
        return self._get('/version').json()['version']

    @property
    @log_args_ret()
    def minimum_compatible_version(self) -> str:
        '''Get the minimum compatible version of the running Rasa server.'''
        return self._get('/version').json()['minimum_compatible_version']

    @property
    @log_args_ret()
    def status(self) -> Dict:
        '''Get the status of the currently loaded model.'''
        return self._get('/status').json()

    @log_args_ret()
    def get_tracker(
        self,
        conversation_id: str,
        include_events:
            Optional[Union['AFTER_RESTART', 'ALL', 'APPLIED', 'NONE']] = None,
        until: Optional[int] = None
    ) -> Dict:
        '''Get the traker of the conversation given by the ID.

        Arguments:
        conversation_id -- ID of the conversation of which to get the tracker

        Keyword Arguments:
        include_events -- Specify which events of the tracker the response
            should contain.
            Can be one of "AFTER_RESTART", "ALL", "APPLIED", "NONE"
            (default: "AFTER_RESTART")
        until -- All events previous to the passed timestamp will be replayed.
            Events that occur exactly at the target time will be included.
            (default: "None")
        '''
        params = {}
        if include_events is not None:
            params['include_events'] = include_events
        if until is not None:
            params['until'] = until
        kwargs = {'params': params} if params else {}
        return self._get(
            f'/conversations/{conversation_id}/tracker', **kwargs
        ).json()

    @log_args_ret()
    def append_event(
        self,
        conversation_id: str,
        event: str,
        timestamp: Optional[int] = None,
        include_events:
            Optional[Union['AFTER_RESTART', 'ALL', 'APPLIED', 'NONE']] = None
    ) -> Dict:
        '''Append a new event to the tracker state of the conversations.

        Arguments:
        conversation_id -- ID of the conversation of which to get the tracker
        event -- event to append to the tracker

        Keyword Arguments:
        timestamp -- Time of application (default: None)
        include_events -- Specify which events of the tracker the response
            should contain.
            Can be one of "AFTER_RESTART", "ALL", "APPLIED", "NONE"
            (default: "AFTER_RESTART")
        '''
        json_ = {'event': event}
        if timestamp is not None:
            json_['timestamp'] = timestamp
        kwargs = {'json': json_}
        if include_events is not None:
            kwargs['params'] = {'include_events': include_events}
        return self._post(
            f'/conversations/{conversation_id}/tracker/events', **kwargs
        ).json()

    @log_args_ret(logging.DEBUG)
    def _handle_events(
        self,
        method: Union['self._post', 'self._put'],
        conversation_id: str,
        events: Iterable[Event],
        include_events:
            Optional[Union['AFTER_RESTART', 'ALL', 'APPLIED', 'NONE']] = None
    ) -> Dict:
        kwargs = {'json': [e.asdict() for e in events]}
        if include_events is not None:
            kwargs['params'] = {'include_events': include_events}
        return method(
            f'/conversations/{conversation_id}/tracker/events', **kwargs
        ).json()

    @log_args_ret()
    def append_events(
        self,
        conversation_id: str,
        events: Iterable[Event],
        include_events:
            Optional[Union['AFTER_RESTART', 'ALL', 'APPLIED', 'NONE']] = None
    ) -> Dict:
        '''Append a new event to the tracker state of the conversations.

        Arguments:
        conversation_id -- ID of the conversation of which to get the tracker
        event -- event to append to the tracker

        Keyword Arguments:
        timestamp -- Time of application (default: None)
        include_events -- Specify which events of the tracker the response
            should contain.
            Can be one of "AFTER_RESTART", "ALL", "APPLIED", "NONE"
            (default: "AFTER_RESTART")
        '''
        return self._handle_events(
            self._post, conversation_id, events, include_events)

    @log_args_ret()
    def replace_events(
        self,
        conversation_id: str,
        events: Iterable[Event],
        include_events:
            Optional[Union['AFTER_RESTART', 'ALL', 'APPLIED', 'NONE']] = None
    ) -> Dict:
        '''Replaces all events of a tracker with the provided list of events.

        Arguments:
        conversation_id -- ID of the conversation of which to get the tracker
        events -- Events which replace the old ones on the tracker

        Keyword Arguments:
        include_events -- Specify which events of the tracker the response
            should contain.
            Can be one of "AFTER_RESTART", "ALL", "APPLIED", "NONE"
            (default: "AFTER_RESTART")
        '''
        return self._handle_events(
            self._put, conversation_id, events, include_events)

    @log_args_ret()
    def get_story(
        self,
        conversation_id: str,
        until: Optional[int] = None
    ) -> Text:
        '''Get an end-to-end story corresponding to a conversation in markdown.

        Arguments:
        conversation_id -- ID of the conversation of which to get the tracker

        Keyword Arguments:
        until -- All events previous to the passed timestamp will be replayed.
            Events that occur exactly at the target time will be included.
            (default: "None")
        '''
        kwargs = {'params': {'until': until}} if until is not None else {}
        return self._get(
            f'/conversations/{conversation_id}/story', **kwargs).text

    @log_args_ret()
    def execute_action(
        self,
        conversation_id: str,
        name: str,
        policy: Optional[str] = None,
        confidence: Optional[float] = None,
        include_events:
            Optional[Union['AFTER_RESTART', 'ALL', 'APPLIED', 'NONE']] = None
    ) -> Dict:
        '''Runs the provided action.

        Arguments:
        conversation_id -- ID of the conversation
        name -- Action name

        Keyword Arguments:
        policy -- Name of the policy that predicted the action
        confidence -- Confidence of the prediction
        include_events -- Specify which events of the tracker the response
            should contain.
            Can be one of "AFTER_RESTART", "ALL", "APPLIED", "NONE"
            (default: "AFTER_RESTART")
        '''
        kwargs = {'params': {'include_events': include_events}} \
            if include_events is not None else {}
        json_ = {'name': name}
        if policy is not None:
            json_['policy'] = policy
        if confidence is not None:
            json_['confidence'] = confidence
        kwargs['json'] = json_
        return self._post(
            f'/conversations/{conversation_id}/execute', **kwargs
        ).json()

    @log_args_ret()
    def score_actions(self, conversation_id: str) -> Dict:
        '''Runs the conversations tracker through the model's policies to
        predict the scores of all actions present in the model's domain.

        Arguments:
        conversation_id -- ID of the conversation of which to get the tracker
        '''
        return self._post(f'/conversations/{conversation_id}/predict').json()

    @log_args_ret()
    def add_message(
        self,
        conversation_id: str,
        text: str,
        sender: str,
        parse_data: Optional['ParseResult'] = None,
        include_events:
            Optional[Union['AFTER_RESTART', 'ALL', 'APPLIED', 'NONE']] = None
    ) -> Dict:
        '''Adds a message to a tracker. This doesn't trigger the prediction
        loop. It will log the message on the tracker and return, no actions
        will be predicted or run. This is often used together with the predict
        endpoint.

        Arguments:
        conversation_id -- the conversations ID
        text -- Message text
        sender -- Origin of the message - who sent it

        Keyword Arguments:
        parse_data -- NLU parser information. If set, message will not be
            passed through NLU, but instead this parsing information will be
            used. (default: None)
        include_events -- Specify which events of the tracker the response
            should contain.
            Can be one of "AFTER_RESTART", "ALL", "APPLIED", "NONE"
            (default: "AFTER_RESTART")

        '''
        kwargs = {'params': {'include_events': include_events}} \
            if include_events else {}
        json_ = {'text': text, 'sender': sender}
        if parse_data is not None:
            json_['parse_data'] = parse_data
        kwargs['json'] = json_
        return self._post(
            f'/conversations/{conversation_id}/messages', **kwargs).json()

    @log_args_ret()
    def train_model(
        self,
        config: str,
        domain: Optional[str] = None,
        nlu: Optional[str] = None,
        stories: Optional[str] = None,
        out: Optional[str] = None,
        force: Optional[bool] = None
    ):
        '''Trains a Rasa model. Depending on the data given only a dialogue
        model, only a NLU model, or a model combining a trained dialogue model
        with an NLU model will be trained. The trained model is not loaded by
        default.

        Arguments:
        config -- Rasa config in plain text

        Keyword arguments:
        domain -- Rasa domain in plain text
        nlu -- Rasa NLU training data in markdown format
        stories -- Rasa Core stories in markdown format
        out -- Output directory
        force -- Force a model training even if the data has not changed
        '''
        json_ = {'config': config}
        if domain is not None:
            json_['domain'] = domain
        if nlu is not None:
            json_['nlu'] = nlu
        if stories is not None:
            json_['stories'] = stories
        if out is not None:
            json_['out'] = out
        if force is not None:
            json_['force'] = force
        return self._post('/model/train', json=json_).json()

    @log_args_ret()
    def evaluate_stories(
        self,
        stories: str,
        e2e: Optional[bool] = None
    ) -> Dict:
        '''Evaluates one or multiple stories against the currently loaded Rasa
        model.

        Arguments:
        stories -- Rasa Core stories in markdown format

        Keyword arguments:
        e2e -- Perform an end-to-end evaluation on the posted stories
            (default: False)
        '''
        kwargs = {'data': stories}
        if e2e is not None:
            kwargs['params'] = {'e2e': 'true' if e2e else 'false'}
        return self._post('/model/test/stories', **kwargs).json()

    @log_args_ret()
    def evaluate_intents(
        self,
        nlu_train_data: str,
        model: Optional[str] = None
    ) -> Dict:
        '''Evaluates intents against the currently loaded Rasa model or the
        model specified in the query.

        Arguments:
        nlu_train_data -- Rasa NLU training data in markdown format

        Keyword Arguements:
        model -- Model that should be used for evaluation
            (example: "rasa-model.tar.gz")
        '''
        kwargs = {'data': nlu_train_data}
        if model is not None:
            kwargs['params'] = {'model': model}
        return self._post('/model/test/intents', **kwargs).json()

    @log_args_ret()
    def predict_action(
        self,
        events: Iterable[Event],
        include_events:
            Optional[Union['AFTER_RESTART', 'ALL', 'APPLIED', 'NONE']] = None
    ) -> Dict:
        '''Predicts the next action on the tracker state as it is posted to
        this endpoint. Rasa will create a temporary tracker from the provided
        events and will use it to predict an action. No messages will be sent
        and no action will be run.

        Arguments:
        events -- Iterable of events to construct temporary tracker from

        Keyword arguments:
        include_events -- Specify which events of the tracker the response
            should contain.
            Can be one of "AFTER_RESTART", "ALL", "APPLIED", "NONE"
            (default: "AFTER_RESTART")
        '''
        kwargs = {'params': {'include_events': include_events}} \
            if include_events is not None else {}
        kwargs['json'] = [e.asdict() for e in events]
        return self._post('/model/predict', **kwargs).json()

    @log_args_ret()
    def parse_message(
        self,
        text: str,
        message_id: Optional[str] = None,
        emulation_mode: Optional[Union['WIT', 'LUIS', 'DIALOGFLOW']] = None
    ) -> Dict:
        '''Predicts the intent and entities of the message posted to this
        endpoint. No messages will be stored to a conversation and no action
        will be run. This will just retrieve the NLU parse results.

        Arguments:
        text -- Message to be parsed

        Keyword arguments:
        message_id -- Optional ID for message to be parsed
        emulation_mode -- Emulation mode of the parsing (default: "LUIS")
        '''
        kwargs = {'params': {'emulation_mode': emulation_mode}} \
            if emulation_mode is not None else {}
        json_ = {'text': text}
        if message_id is not None:
            json_['message_id'] = json_
        kwargs['json'] = json_
        return self._post('/model/parse', **kwargs).json()

    @log_args_ret()
    def replace_current_model(
        self,
        model_file: Optional[str] = None,
        model_server: Optional['EndPointConfig'] = None,
        remote_storage: Optional[Union['aws', 'gcs', 'azure']] = None
    ) -> Dict:
        '''Updates the currently loaded model. First, tries to load the model
        from the local storage system. Secondly, tries to load the model from
        the provided model server configuration. Last, tries to load the model
        from the provided remote storage.

        Keyword arguments:
        model_file -- Path to model file
        model_server -- EndpointConfig pointing on server with the model on it
        remote_storage -- Name of the used remote storage.
            Can be one of "aws", "gcs" or "azure"
        '''
        json_ = {}
        if model_file is not None:
            json_['model_file'] = model_file
        if model_server is not None:
            json_['model_server'] = model_server
        if remote_storage is not None:
            json_['remote_storage'] = remote_storage
        return self._put('/model', json=json_).json()

    @log_args_ret()
    def unload_current_model(self) -> Dict:
        '''Unloads the currently loaded trained model from the server.'''
        return self._delete('/model').json()

    @log_args_ret()
    def domain(self,
               type_: Union['yaml', 'json'] = 'yaml') -> Union[str, Dict]:
        '''Returns the domain specification the currently loaded model is using.

        Keyword arguments:
        type_ -- Accept header type.
            Can be either "yaml" or "json" (default: "yaml")
        '''
        if type_ not in ('yaml', 'json'):
            raise ValueError(
                'type_ parameter has to be either "yaml" or "json".')
        resp = self._get('/domain', headers={"Accept": f"application/{type_}"})
        return resp.text if type_ == 'yaml' else resp.json()


def int_test():
    import sys
    url = sys.argv[1]
    token = sys.argv[2]
    rpi = RasaPI(url, token)

    rpi.version
    rpi.minimum_compatible_version
    rpi.status
    rpi.get_tracker('test')
    rpi.append_event('test', 'slot')
    rpi.append_events('test', [Event('slot'), Event('slot', 1234)])
    rpi.replace_events('test', [Event('slot'), Event('slot', 1234)])
    rpi.get_story('test')
    rpi.execute_action('test', 'utter_greet')
    rpi.score_actions('test')
    rpi.add_message('test', 'hello', 'testuser', {})
    rpi.train_model(
        **{'config': 'wurst', 'domain': 'a', 'nlu': 'b', 'stories': 'c',
           'out': 'd', 'force': False}
    )
    rpi.evaluate_stories('MARKDOWN')
    rpi.evaluate_intents('MARKDOWN', 'model.tar.gz')
    rpi.predict_action([Event('slot'), Event('slot', 1234)])
    rpi.parse_message('message')
    rpi.replace_current_model('/model.tar.gz')
    rpi.unload_current_model()
    rpi.domain()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    int_test()
