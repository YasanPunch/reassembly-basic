import uuid
from abc import ABC, abstractmethod
import open3d.visualization.gui as gui  # type: ignore


class BaseItem(ABC):
    """
    Base class for all items in the item tree.
    Defines common properties and methods that all items must have.
    """
    
    def __init__(self, label: str, is_visible: bool = True):
        """
        Initialize a base item.
        
        Args:
            label (str): Display label for the item
            is_visible (bool): Whether the item is visible by default
        """
        self._id = str(uuid.uuid4())  # Unique identifier
        self._label = label
        self._is_visible = is_visible
        self._ui_widget = None  # GUI widget associated with this item
        
    @property
    def id(self) -> str:
        """Get the unique identifier of the item."""
        return self._id
    
    @property
    def label(self) -> str:
        """Get the display label of the item."""
        return self._label
    
    @label.setter
    def label(self, value: str):
        """Set the display label of the item."""
        self._label = value
        if self._ui_widget:
            self._update_ui_label()
    
    @property
    def is_visible(self) -> bool:
        """Get whether the item is visible."""
        return self._is_visible
    
    @is_visible.setter
    def is_visible(self, value: bool):
        """Set whether the item is visible."""
        self._is_visible = value
        if self._ui_widget:
            self._update_ui_visibility()
    
    @property
    def is_hidden(self) -> bool:
        """Get whether the item is hidden (inverse of is_visible)."""
        return not self._is_visible
    
    @is_hidden.setter
    def is_hidden(self, value: bool):
        """Set whether the item is hidden (inverse of is_visible)."""
        self.is_visible = not value
    
    def set_ui_widget(self, widget):
        """Set the UI widget associated with this item."""
        self._ui_widget = widget
        self._update_ui_label()
        self._update_ui_visibility()
    
    def _update_ui_label(self):
        """Update the UI widget label. Override in subclasses if needed."""
        if self._ui_widget and hasattr(self._ui_widget, 'text'):
            self._ui_widget.text = self._label
    
    def _update_ui_visibility(self):
        """Update the UI widget visibility. Override in subclasses if needed."""
        if self._ui_widget and hasattr(self._ui_widget, 'checked'):
            self._ui_widget.checked = self._is_visible
    
    @abstractmethod
    def get_item_type(self) -> str:
        """Get the type of this item. Must be implemented by subclasses."""
        pass
    
    def to_dict(self) -> dict:
        """Convert the item to a dictionary representation."""
        return {
            'id': self._id,
            'label': self._label,
            'is_visible': self._is_visible,
            'type': self.get_item_type()
        }
    
    def __str__(self) -> str:
        return f"{self.get_item_type()}(id={self._id}, label='{self._label}', visible={self._is_visible})"
    
    def __repr__(self) -> str:
        return self.__str__() 