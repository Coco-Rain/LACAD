import cadquery as cq

result = (
cq.Workplane("XZ")
.tag('base')
.box(100, 95, 13, centered=(True, True, False))
.workplaneFromTagged('base')
.workplane(offset=13)
.tag('holeplane')
.rect(80, 34, forConstruction=True)
.vertices()
.hole(8)
)
cq.exporters.export(result, 'GT.stl')